import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import serial
import time
import threading
from dataclasses import dataclass
from typing import Optional

@dataclass
class PredictionResult:
    """Container for prediction results"""
    digit: Optional[int] = None
    error: Optional[str] = None

class DrawingCanvas:
    """Canvas for drawing digits"""
    def __init__(self, parent, size=320):
        self.size = size
        self.canvas = tk.Canvas(parent, width=size, height=size, bg='white', 
                                cursor='crosshair', highlightthickness=2, 
                                highlightbackground='#e5e7eb', relief='flat')
        self.canvas.pack(pady=15)
        
        # Create PIL image for drawing
        self.image = Image.new('L', (size, size), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Drawing state
        self.last_x = None
        self.last_y = None
        self.is_drawing = False
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
    
    def start_draw(self, event):
        """Start drawing"""
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_line(self, event):
        """Draw line on canvas"""
        if self.is_drawing:
            x, y = event.x, event.y
            
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=8, fill='#1e293b', capstyle=tk.ROUND, smooth=True
            )
            
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill='black', width=24
            )
            
            self.last_x = x
            self.last_y = y
    
    def stop_draw(self, event):
        """Stop drawing"""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
    
    def clear(self):
        """Clear the canvas"""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.size, self.size), 'white')
        self.draw = ImageDraw.Draw(self.image)
    
    def get_image_array(self):
        """Get preprocessed image as numpy array"""
        # Resize to 28x28 (MNIST size)
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Invert colors (white digit on black background)
        img_array = np.array(img_resized, dtype=np.uint8)
        img_array = 255 - img_array
        
        return img_array.flatten()

class LoadingSpinner:
    """Animated loading spinner"""
    def __init__(self, parent, size=40):
        self.canvas = tk.Canvas(parent, width=size, height=size, 
                               bg='white', highlightthickness=0)
        self.size = size
        self.angle = 0
        self.is_running = False
        self.animation_id = None
        
    def pack(self, **kwargs):
        self.canvas.pack(**kwargs)
        
    def pack_forget(self):
        self.stop()
        self.canvas.pack_forget()
        
    def start(self):
        """Start spinner animation"""
        self.is_running = True
        self.animate()
        
    def stop(self):
        """Stop spinner animation"""
        self.is_running = False
        if self.animation_id:
            self.canvas.after_cancel(self.animation_id)
            self.animation_id = None
        self.canvas.delete('all')
        
    def animate(self):
        """Animate the spinner"""
        if not self.is_running:
            return
            
        self.canvas.delete('all')
        center = self.size // 2
        radius = self.size // 3
        
        # Draw arc
        for i in range(8):
            angle = (self.angle + i * 45) % 360
            opacity = int(255 * (i / 8))
            color = f'#{opacity:02x}{opacity:02x}{opacity:02x}'
            
            x = center + radius * np.cos(np.radians(angle))
            y = center + radius * np.sin(np.radians(angle))
            
            self.canvas.create_oval(x-3, y-3, x+3, y+3, 
                                   fill='#3b82f6', outline='')
        
        self.angle = (self.angle + 10) % 360
        self.animation_id = self.canvas.after(50, self.animate)

class STM32DigitClassifier:
    """Main application window"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("STM32 Digit Classifier")
        self.root.geometry("750x700")
        self.root.minsize(600, 600)
        self.root.resizable(True, True)
        self.root.configure(bg='#f8fafc')
        
        # Serial connection
        self.serial_conn = None
        self.is_connected = False
        
        # Screens
        self.screens = {}
        self.current_screen = None
        
        # Configure style
        self.setup_style()
        
        # Setup GUI
        self.setup_ui()
        
    def setup_style(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Button styles with better shadows
        style.configure('Primary.TButton', 
                       background='#3b82f6', 
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 12),
                       font=('Segoe UI', 10, 'bold'))
        style.map('Primary.TButton',
                 background=[('active', '#2563eb'), ('disabled', '#93c5fd')])
        
        style.configure('Secondary.TButton',
                       background='#64748b',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 12),
                       font=('Segoe UI', 10, 'bold'))
        style.map('Secondary.TButton',
                 background=[('active', '#475569')])
        
        style.configure('Danger.TButton',
                       background='#ef4444',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 12),
                       font=('Segoe UI', 10, 'bold'))
        style.map('Danger.TButton',
                 background=[('active', '#dc2626')])
        
        style.configure('Success.TButton',
                       background='#10b981',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(20, 12),
                       font=('Segoe UI', 10, 'bold'))
        style.map('Success.TButton',
                 background=[('active', '#059669')])
        
        # Entry style
        style.configure('TEntry',
                       fieldbackground='white',
                       borderwidth=2,
                       relief='flat',
                       padding=10)
        
        # Progressbar style
        style.configure('TProgressbar',
                       background='#3b82f6',
                       troughcolor='#e2e8f0',
                       borderwidth=0,
                       thickness=6)
        
    def setup_ui(self):
        """Setup the user interface"""
        
        # Enhanced header with gradient effect
        self.title_frame = tk.Frame(self.root, bg='#1e40af', height=120)
        self.title_frame.pack(fill=tk.X)
        self.title_frame.pack_propagate(False)
        
        # Create gradient effect with overlays
        gradient_overlay = tk.Frame(self.title_frame, bg='#1d4ed8', height=40)
        gradient_overlay.place(x=0, y=80, relwidth=1)
        
        title_label = tk.Label(
            self.title_frame, text="🧠 STM32 Digit Classifier",
            font=('Segoe UI', 28, 'bold'), fg='white', bg='#1e40af'
        )
        title_label.pack(pady=(25, 5))
        
        subtitle_label = tk.Label(
            self.title_frame, text="Neural Network on Microcontroller · Real-time Inference",
            font=('Segoe UI', 10), fg='#bfdbfe', bg='#1e40af'
        )
        subtitle_label.pack()
        
        self.setup_connection_screen()
        self.setup_drawing_screen()
        self.setup_result_screen()
        
        self.show_screen('connection')
        
    def setup_connection_screen(self):
        """Setup connection screen"""
        self.connection_screen = tk.Frame(self.root, bg='#f8fafc')
        
        # Spacer for centering
        spacer_top = tk.Frame(self.connection_screen, bg='#f8fafc')
        spacer_top.pack(fill=tk.BOTH, expand=True)
        
        # Card with shadow effect
        card_shadow = tk.Frame(self.connection_screen, bg='#cbd5e1', bd=0)
        card_shadow.pack(padx=40, pady=20)
        
        conn_frame = tk.Frame(card_shadow, bg='white', bd=0, relief='flat')
        conn_frame.pack(padx=1, pady=1, fill=tk.BOTH, expand=True)
        
        # Spacer for centering
        spacer_bottom = tk.Frame(self.connection_screen, bg='#f8fafc')
        spacer_bottom.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(conn_frame, bg='#f8fafc', height=60)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame, text="Serial Connection",
            font=('Segoe UI', 16, 'bold'), fg='#1e293b', bg='#f8fafc'
        ).pack(pady=15)
        
        # Content area
        content_frame = tk.Frame(conn_frame, bg='white')
        content_frame.pack(fill=tk.BOTH, padx=30, pady=20)
        
        # Port configuration
        config_frame = tk.Frame(content_frame, bg='#f8fafc', bd=0)
        config_frame.pack(fill=tk.X, pady=(0, 20), ipady=15, ipadx=15)
        
        # Port
        port_container = tk.Frame(config_frame, bg='#f8fafc')
        port_container.pack(fill=tk.X, pady=8)
        
        tk.Label(
            port_container, text="Port", width=10, anchor='w',
            font=('Segoe UI', 10, 'bold'), bg='#f8fafc', fg='#475569'
        ).pack(side=tk.LEFT)
        
        self.port_var = tk.StringVar(value='COM9')
        port_entry = ttk.Entry(port_container, textvariable=self.port_var, 
                              width=25, font=('Segoe UI', 10))
        port_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Baud rate
        baud_container = tk.Frame(config_frame, bg='#f8fafc')
        baud_container.pack(fill=tk.X, pady=8)
        
        tk.Label(
            baud_container, text="Baud Rate", width=10, anchor='w',
            font=('Segoe UI', 10, 'bold'), bg='#f8fafc', fg='#475569'
        ).pack(side=tk.LEFT)
        
        self.baud_var = tk.StringVar(value='115200')
        baud_entry = ttk.Entry(baud_container, textvariable=self.baud_var, 
                              width=25, font=('Segoe UI', 10))
        baud_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Loading spinner (hidden initially)
        self.connection_spinner = LoadingSpinner(content_frame, size=40)
        
        # Connect button
        self.connect_btn = ttk.Button(
            content_frame, text="🔌 Connect to Device", command=self.connect,
            style='Primary.TButton'
        )
        self.connect_btn.pack(pady=15)
        
        # Status indicator
        status_container = tk.Frame(content_frame, bg='#f1f5f9', bd=0, 
                                   relief='flat')
        status_container.pack(pady=15, fill=tk.X, ipady=12)
        
        self.status_label = tk.Label(
            status_container, text="● Disconnected", 
            fg='#dc2626', font=('Segoe UI', 11, 'bold'),
            bg='#f1f5f9'
        )
        self.status_label.pack()
        
        # Next button
        self.next_btn = ttk.Button(
            content_frame, text="Continue to Drawing →", 
            command=lambda: self.show_screen('drawing'),
            state='disabled',
            style='Success.TButton'
        )
        self.next_btn.pack(pady=10)
        
    
        self.screens['connection'] = self.connection_screen
    
    def setup_drawing_screen(self):
        """Setup drawing screen"""
        self.drawing_screen = tk.Frame(self.root, bg='#f8fafc')
        
        # Spacer for centering
        spacer_top = tk.Frame(self.drawing_screen, bg='#f8fafc')
        spacer_top.pack(fill=tk.BOTH, expand=True)
        
        # Card shadow
        card_shadow = tk.Frame(self.drawing_screen, bg='#cbd5e1', bd=0)
        card_shadow.pack(padx=40, pady=20)
        
        draw_frame = tk.Frame(card_shadow, bg='white', bd=0)
        draw_frame.pack(padx=1, pady=1, fill=tk.BOTH, expand=True)
        
        # Spacer for centering
        spacer_bottom = tk.Frame(self.drawing_screen, bg='#f8fafc')
        spacer_bottom.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(draw_frame, bg='#f8fafc', height=60)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame, text="Draw a Digit (0-9)",
            font=('Segoe UI', 16, 'bold'), fg='#1e293b', bg='#f8fafc'
        ).pack(pady=15)
        
        # Content
        content_frame = tk.Frame(draw_frame, bg='white')
        content_frame.pack(fill=tk.BOTH, padx=30, pady=20)
        
        self.canvas = DrawingCanvas(content_frame)
        
        # Button frame
        btn_frame = tk.Frame(content_frame, bg='white')
        btn_frame.pack(pady=15)
        
        ttk.Button(
            btn_frame, text="🗑 Clear Canvas", command=self.clear_canvas, 
            width=18, style='Danger.TButton'
        ).pack(side=tk.LEFT, padx=5)
        
        self.send_btn = ttk.Button(
            btn_frame, text="🚀 Predict Digit", command=self.classify_digit,
            width=18, style='Primary.TButton'
        )
        self.send_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame, text="⏏ Disconnect", command=self.disconnect_and_back, 
            width=18, style='Secondary.TButton'
        ).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        progress_container = tk.Frame(content_frame, bg='white')
        progress_container.pack(pady=10)
        
        self.progress = ttk.Progressbar(
            progress_container, mode='indeterminate', length=500, 
            style='TProgressbar'
        )
        
        self.progress_label = tk.Label(
            progress_container, text="Processing...",
            font=('Segoe UI', 9), fg='#64748b', bg='white'
        )
        
        self.screens['drawing'] = self.drawing_screen
    
    def setup_result_screen(self):
        """Setup result screen"""
        self.result_screen = tk.Frame(self.root, bg='#f8fafc')
        
        # Spacer for centering
        spacer_top = tk.Frame(self.result_screen, bg='#f8fafc')
        spacer_top.pack(fill=tk.BOTH, expand=True)
        
        # Card shadow
        card_shadow = tk.Frame(self.result_screen, bg='#cbd5e1', bd=0)
        card_shadow.pack(padx=40, pady=20)
        
        result_frame = tk.Frame(card_shadow, bg='white', bd=0)
        result_frame.pack(padx=1, pady=1, fill=tk.BOTH, expand=True)
        
        # Spacer for centering
        spacer_bottom = tk.Frame(self.result_screen, bg='#f8fafc')
        spacer_bottom.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(result_frame, bg='#f8fafc', height=60)
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame, text="Prediction Result",
            font=('Segoe UI', 16, 'bold'), fg='#1e293b', bg='#f8fafc'
        ).pack(pady=15)
        
        # Content
        content_frame = tk.Frame(result_frame, bg='white')
        content_frame.pack(fill=tk.BOTH, padx=30, pady=20)
        
        # Result display
        self.result_container = tk.Frame(content_frame, bg='#f8fafc', 
                                        relief='flat', bd=0)
        self.result_container.pack(pady=15, fill=tk.X, ipady=20)
        
        self.result_text = tk.Label(
            self.result_container, text="",
            font=('Segoe UI', 16), fg='#64748b', bg='#f8fafc'
        )
        self.result_text.pack(pady=15)
        
        # Button frame
        btn_frame = tk.Frame(content_frame, bg='white')
        btn_frame.pack(pady=20)
        
        ttk.Button(
            btn_frame, text="✏ Draw Another Digit", command=self.draw_another, 
            width=22, style='Success.TButton'
        ).pack(side=tk.LEFT, padx=8)
        
        ttk.Button(
            btn_frame, text="⏏ Disconnect Device", command=self.disconnect_and_back, 
            width=22, style='Secondary.TButton'
        ).pack(side=tk.LEFT, padx=8)
        
        self.screens['result'] = self.result_screen
    
    def show_screen(self, screen_name):
        """Switch to specified screen"""
        if self.current_screen:
            self.current_screen.pack_forget()
        
        screen = self.screens[screen_name]
        screen.pack(fill=tk.BOTH, expand=True)
        self.current_screen = screen
    
    def connect(self):
        """Connect to STM32"""
        # Show loading state
        self.connect_btn.config(state='disabled', text="Connecting...")
        self.connection_spinner.pack(pady=15)
        self.connection_spinner.start()
        self.status_label.config(text="● Connecting...", fg='#f59e0b')
        
        # Run connection in thread
        thread = threading.Thread(target=self.connect_thread)
        thread.daemon = True
        thread.start()
    
    def connect_thread(self):
        """Connection thread"""
        port = self.port_var.get()
        baud = int(self.baud_var.get())
        
        try:
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=5,
                write_timeout=5
            )
            time.sleep(2)  # Wait for connection
            
            # Clear buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # Read banner
            self.read_banner()
            
            self.is_connected = True
            
            # Update UI in main thread
            self.root.after(0, self.on_connect_success)
            
        except Exception as e:
            # Update UI in main thread
            self.root.after(0, lambda: self.on_connect_error(str(e)))
    
    def on_connect_success(self):
        """Handle successful connection"""
        self.connection_spinner.stop()
        self.connection_spinner.pack_forget()
        self.status_label.config(text="✓ Connected Successfully", fg='#10b981')
        self.connect_btn.config(text="🔌 Connect to Device")
        self.next_btn.config(state='normal')
    
    def on_connect_error(self, error_msg):
        """Handle connection error"""
        self.connection_spinner.stop()
        self.connection_spinner.pack_forget()
        self.connect_btn.config(state='normal', text="🔌 Connect to Device")
        self.status_label.config(text="● Connection Failed", fg='#dc2626')
        messagebox.showerror("Connection Error", f"Failed to connect:\n\n{error_msg}")
    
    def disconnect(self):
        """Disconnect from STM32"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        
        self.is_connected = False
        self.serial_conn = None
    
    def disconnect_and_back(self):
        """Disconnect and return to connection screen"""
        self.disconnect()
        self.status_label.config(text="● Disconnected", fg='#dc2626')
        self.connect_btn.config(state='normal')
        self.next_btn.config(state='disabled')
        self.show_screen('connection')
    
    def read_banner(self):
        """Read initialization banner"""
        start = time.time()
        while time.time() - start < 2:
            if self.serial_conn.in_waiting:
                self.serial_conn.readline()
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.clear()
    
    def draw_another(self):
        """Clear canvas and return to drawing screen"""
        self.clear_canvas()
        self.show_screen('drawing')
    
    def classify_digit(self):
        """Classify the drawn digit"""
        if not self.is_connected:
            messagebox.showwarning("Not Connected", "Please connect to STM32 first")
            return
        
        # Disable send button and show progress
        self.send_btn.config(state='disabled')
        self.progress.pack()
        self.progress.start(10)
        self.progress_label.pack(pady=(10, 0))
        
        # Run classification in separate thread
        thread = threading.Thread(target=self.classify_thread)
        thread.daemon = True
        thread.start()
    
    def classify_thread(self):
        """Classification thread"""
        try:
            # Get image data
            img_data = self.canvas.get_image_array()
            
            # Send to STM32
            result = self.send_and_receive(img_data)
            
            # Update UI in main thread
            self.root.after(0, lambda: self.display_result(result))
            
        except Exception as e:
            error_result = PredictionResult(error=str(e))
            self.root.after(0, lambda: self.display_result(error_result))
    
    def send_and_receive(self, img_data):
        """Send image and receive prediction"""
        result = PredictionResult()
        
        try:
            # Clear buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # Send START command
            self.serial_conn.write(b"START")
            self.serial_conn.flush()
            time.sleep(0.05)
            
            # Send image data in chunks
            CHUNK = 64
            for i in range(0, len(img_data), CHUNK):
                chunk = img_data[i:i + CHUNK]
                self.serial_conn.write(chunk.tobytes())
                self.serial_conn.flush()
                time.sleep(0.005)
            
            # Read response
            start = time.time()
            timeout = 10
            all_lines = []
            
            while time.time() - start < timeout:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode(errors='ignore').strip()
                    
                    if not line:
                        continue
                    
                    all_lines.append(line)
                    
                    # Check for error
                    if "ERROR" in line.upper():
                        result.error = line
                        return result
                    
                    # Parse digit
                    try:
                        digit = int(line)
                        result.digit = digit
                        return result
                    except ValueError:
                        pass
                
                time.sleep(0.01)
            
            # Timeout
            if all_lines:
                result.error = f"No digit found. Received: {', '.join(all_lines[:5])}"
            else:
                result.error = "Timeout: No response from STM32"
            
        except Exception as e:
            result.error = f"Communication error: {str(e)}"
        
        return result
    
    def display_result(self, result: PredictionResult):
        """Display classification result"""
        self.progress.stop()
        self.progress.pack_forget()
        self.progress_label.pack_forget()
        self.send_btn.config(state='normal')
        
        if result.error:
            self.result_container.config(bg='#fef2f2')
            self.result_text.config(
                text=f"❌ Error\n\n{result.error}",
                fg='#dc2626',
                font=('Segoe UI', 12),
                bg='#fef2f2'
            )
        elif result.digit is not None:
            self.result_container.config(bg='#f0fdf4')
            result_str = f"🎯 Predicted Digit\n\n{result.digit}"
            
            self.result_text.config(
                text=result_str,
                fg='#10b981',
                font=('Segoe UI', 42, 'bold'),
                bg='#f0fdf4'
            )
            self.root.update_idletasks()
        else:
            self.result_container.config(bg='#fef2f2')
            self.result_text.config(
                text="❌ No prediction received",
                fg='#dc2626',
                font=('Segoe UI', 12),
                bg='#fef2f2'
            )
        
        # Switch to result screen
        self.show_screen('result')
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_connected:
            self.disconnect()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = STM32DigitClassifier(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()