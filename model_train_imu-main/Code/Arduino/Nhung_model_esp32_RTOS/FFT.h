#ifndef FFT_H
#define FFT_H

const int resp_samples = 512;
float resp_data[resp_samples] = {};
float frequencies[resp_samples];
float magnitudes[resp_samples];
float real_part[resp_samples];
float imag_part[resp_samples];
float respiratory = 0; 

void fft(float *real, float *imag, int N);

//Fourier Transform
void compute_fourier_transform(float* data, float* frequencies, float* magnitudes, int data_length) 
{
  Serial.println();
  Serial.println("Computing Fourier Transform...");

  // Remove DC
  float mean_value = 0;
  for (int i = 0; i < data_length; i++) 
  {
    mean_value += data[i];
  }
  mean_value /= data_length;

  for (int i = 0; i < data_length; i++) 
  {
    data[i] -= mean_value;
  }

  // Calculate real part and imag part
  for (int k = 0; k < data_length; k++) 
  {
    real_part[k] = data[k];
    imag_part[k] = 0;
  }
  
  //FFT
  fft(real_part, imag_part, data_length);

  for (int k = 0; k < data_length; k++) 
  {
    magnitudes[k] = sqrt(real_part[k] * real_part[k] + imag_part[k] * imag_part[k]);
    frequencies[k] = k * 40.0 / data_length; // Frequencies
  }

  Serial.println("Fourier Transform computed.");
}
// Calculate respiratory
int rr_est(float* magnitudes, int samples) 
{
  Serial.println("Estimating breath rate from magnitudes...");
  int max_peak_index = 0;
  float max_peak_value = 0;

  for (int i = 0; i < 7; i++) 
  {
    if (magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1]) 
    {
      if (magnitudes[i] > max_peak_value) 
      {
        max_peak_value = magnitudes[i];
        max_peak_index = i;
      }
    }
  }

  // Convert peak index to respiratory rate
  float breath_rate = (max_peak_index * 40.0 / samples) * 60; // Breathing rate per minute
  Serial.print("Nhịp thở ước lượng: ");
  Serial.println(breath_rate);

  return breath_rate;
}

//FFT algorithm
void fft(float *real, float *imag, int N) {
    int j = 0;
    for (int i = 1; i < N; i++) 
    {
        int bit = N >> 1;
        while (j >= bit) 
        {
            j -= bit;
            bit >>= 1;
        }
        j += bit;

        if (i < j) 
        {
            // Hoán đổi các phần tử
            float temp_real = real[i];
            float temp_imag = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = temp_real;
            imag[j] = temp_imag;
        }
    }

    for (int len = 2; len <= N; len <<= 1) 
    {
        float angle = -2 * PI / len;
        float w_real = cos(angle);
        float w_imag = sin(angle);
        for (int i = 0; i < N; i += len) 
        {
            float u_real = 1;
            float u_imag = 0;
            for (int j = 0; j < len / 2; j++) 
            {
                int index1 = i + j;
                int index2 = i + j + len / 2;

                float temp_real = u_real * real[index2] - u_imag * imag[index2];
                float temp_imag = u_real * imag[index2] + u_imag * real[index2];

                real[index2] = real[index1] - temp_real;
                imag[index2] = imag[index1] - temp_imag;
                real[index1] += temp_real;
                imag[index1] += temp_imag;

                // Update u
                float temp_u_real = u_real * w_real - u_imag * w_imag;
                u_imag = u_real * w_imag + u_imag * w_real;
                u_real = temp_u_real;
            }
        }
    }
}
#endif