/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body with AI Shape Classification (STM32F411)
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"
#include <string.h>
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef enum {
  STATE_IDLE,
  STATE_WAIT_START,
  STATE_RECEIVE_IMAGE,
  STATE_PROCESS_IMAGE
} AppState_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// LED configuration - PD13
#define LED_PIN GPIO_PIN_13
#define LED_GPIO_PORT GPIOD
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
// AI model variables
AI_ALIGNED(4) static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

static ai_handle network = AI_HANDLE_NULL;
static ai_buffer *ai_input;
static ai_buffer *ai_output;

// Image and classification buffers
#define IMG_SIZE 784
#define IMG_WIDTH 28
#define IMG_HEIGHT 28
#define NUM_CLASSES 10

static uint8_t img_buffer[IMG_SIZE];
static int8_t input_buffer[IMG_SIZE];
static int8_t output_buffer[NUM_CLASSES];

// UART interrupt variables
static volatile AppState_t app_state = STATE_IDLE;
static uint8_t start_cmd_buffer[5];
static volatile uint8_t rx_complete = 0;
static volatile uint8_t rx_error = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */
int AI_Init(void);
int AI_Run(int8_t *input_data, int8_t *output_data);
void ProcessInference(void);
void SendResult(int predicted_class);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/**
  * @brief Initialize AI model
  */
int AI_Init(void)
{
  ai_error err;

  const ai_handle act_addr[] = { AI_HANDLE_PTR(activations) };

  err = ai_network_create_and_init(&network, act_addr, NULL);
  if (err.type != AI_ERROR_NONE)
  {
    return -1;
  }

  ai_input = ai_network_inputs_get(network, NULL);
  ai_output = ai_network_outputs_get(network, NULL);

  if (!ai_input || !ai_output)
  {
    return -1;
  }

  return 0;
}

/**
  * @brief Run AI inference
  */
int AI_Run(int8_t *input_data, int8_t *output_data)
{
  ai_i32 batch;

  if (!network || !ai_input || !ai_output) return -1;

  ai_input[0].data = AI_HANDLE_PTR(input_data);
  ai_output[0].data = AI_HANDLE_PTR(output_data);

  batch = ai_network_run(network, ai_input, ai_output);
  if (batch != 1)
  {
    return -1;
  }

  return 0;
}

/**
  * @brief Process inference and control LED
  */
void ProcessInference(void)
{
  // Convert uint8 (0-255) to int8 (-128 to 127)
  for (int i = 0; i < IMG_SIZE; i++)
  {
    input_buffer[i] = (int8_t)(img_buffer[i] - 128);
  }

  // Run inference with timing
  uint32_t start_tick = HAL_GetTick();
  int ret = AI_Run(input_buffer, output_buffer);

  if (ret == 0)
  {
    // Find max class
    int predicted_class = 0;
    int8_t max_prob = output_buffer[0];
    for (int i = 1; i < NUM_CLASSES; i++)
    {
      if (output_buffer[i] > max_prob)
      {
        max_prob = output_buffer[i];
        predicted_class = i;
      }
    }

    // Send result
    SendResult(predicted_class);
  }
  else
  {
    // Error during inference
    char err_msg[48];
    snprintf(err_msg, sizeof(err_msg), "ERROR: Inference failed\r\n");
    HAL_UART_Transmit(&huart2, (uint8_t*)err_msg, strlen(err_msg), 1000);
  }
}

/**
  * @brief Send classification result via UART
  */
void SendResult(int predicted_class)
{
  char result[32];
  snprintf(result, sizeof(result), "%d\r\n", predicted_class);
  HAL_UART_Transmit(&huart2, (uint8_t*)result, strlen(result), 1000);

}


/**
  * @brief UART Rx Complete Callback
  */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    if (app_state == STATE_WAIT_START)
    {
      // Check if START command received
      if (memcmp(start_cmd_buffer, "START", 5) == 0)
      {
        // Start receiving image data
        app_state = STATE_RECEIVE_IMAGE;
        HAL_UART_Receive_IT(&huart2, img_buffer, IMG_SIZE);
      }
      else
      {
        // Not a valid START command, wait again
        HAL_UART_Receive_IT(&huart2, start_cmd_buffer, 5);
      }
    }
    else if (app_state == STATE_RECEIVE_IMAGE)
    {
      // Image reception complete
      rx_complete = 1;
      app_state = STATE_PROCESS_IMAGE;
    }
  }
}

/**
  * @brief UART Error Callback
  */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    rx_error = 1;
    app_state = STATE_IDLE;

    char err_msg[48];
    snprintf(err_msg, sizeof(err_msg), "ERROR: UART error\r\n");
    HAL_UART_Transmit(&huart2, (uint8_t*)err_msg, strlen(err_msg), 1000);
  }
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */

  // Initialize AI model
  if (AI_Init() != 0)
  {
    char msg[64];
    snprintf(msg, sizeof(msg), "AI Init Failed!\r\n");
    HAL_UART_Transmit(&huart2, (uint8_t*)msg, strlen(msg), 1000);
    // Blink LED rapidly to indicate error
    while(1) {}
  }

  // Small delay so host side can open VCP
  HAL_Delay(200);

  // Send ready message
  {
    char ready_msg[64];
    snprintf(ready_msg, sizeof(ready_msg), "STM32F411 Ready - Cube AI Initialized\r\n");
    HAL_UART_Transmit(&huart2, (uint8_t*)ready_msg, strlen(ready_msg), HAL_MAX_DELAY);
  }

  // Start waiting for START command using interrupt
  app_state = STATE_WAIT_START;
  HAL_UART_Receive_IT(&huart2, start_cmd_buffer, 5);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    // Process image when reception is complete
    if (rx_complete)
    {
      rx_complete = 0;
      ProcessInference();

      // Go back to waiting for START command
      app_state = STATE_WAIT_START;
      HAL_UART_Receive_IT(&huart2, start_cmd_buffer, 5);
    }

    // Handle errors
    if (rx_error)
    {
      rx_error = 0;

      // Restart waiting for START command
      app_state = STATE_WAIT_START;
      HAL_UART_Receive_IT(&huart2, start_cmd_buffer, 5);
    }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 192;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 8;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, GPIO_PIN_13, GPIO_PIN_RESET);

  /*Configure GPIO pin : PD13 */
  GPIO_InitStruct.Pin = GPIO_PIN_13;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
