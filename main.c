// ---------- //
// Mikian Musser
// Summer 2019
// ---------- //

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <time.h>
#include <math.h>

// Training
#define EPOCHS 2
#define INTERVALS 10000
#define L      0.2
#define DL     0.001
#define GL     L

// Generator Structure
#define GIN     100
#define GHIDDEN 512
#define GOUT    784

// Discriminator Structure
#define DIN     784
#define DHIDDEN  40
#define DOUT      2

// MNIST Data
#define TRAINING 60000
#define TESTING  10000
#define ROWS        28
#define COLS        28
#define TRIF "data/train-images.idx3-ubyte"
#define TEIF  "data/t10k-images.idx3-ubyte"

// Image Data
double TRP[TRAINING][ROWS][COLS];
double TEP[TESTING][ROWS][COLS];
double TRA[ROWS][COLS];
double TRD[ROWS][COLS];

// Generator Trainable Paramaters
double G1W[GIN][GHIDDEN];
double G1B[GHIDDEN];
double G2W[GHIDDEN][GOUT];
double G2B[GOUT];

// Discriminator Trainable Paramaters
double D1W[DIN][DHIDDEN];
double D1B[DHIDDEN];
double D2W[DHIDDEN][DOUT];
double D2B[DOUT];

// Generator Grads
double G1WGRAD[GIN][GHIDDEN];
double G1BGRAD[GHIDDEN];
double G2WGRAD[GHIDDEN][GOUT];
double G2BGRAD[GOUT];

// Discriminator Grads
double D1WGRAD[DIN][DHIDDEN];
double D1BGRAD[DHIDDEN];
double D2WGRAD[DHIDDEN][DOUT];
double D2BGRAD[DOUT];

// Generator Data
double GINDATA[GIN];
double GHIDDENDATA[GHIDDEN];
double GOUTDATA[GOUT];

// Discriminator data
double DINDATA[DIN];
double DHIDDENDATA[DHIDDEN];
double DOUTDATA[DOUT];

// Network
double OUTPUT[DOUT];
double ANSWER[DOUT];
double ERROR[DOUT];

// Back Prop Storage
double OUTERROR[DOUT];
double HIDDENERROR[DHIDDEN];
double INPUTERROR[GOUT];
double GHIDDENERROR[GOUT];

double XavierRand(double fan_in);
double Sigmoid(double x);
double NormalRand(double m, double sd);

int main() {

  // Print Header
  printf("\n------- Mikian  Musser -------\n\n"     );

  // Create gen vars
  int readInt = 0;
  int interval = 100;

  // Open data files
  FILE * f_trp = fopen(TRIF, "rb");
  FILE * f_tep = fopen(TEIF, "rb");
  if (!f_trp || !f_tep) printf("Error Opening Data Files.\n");

  // Read in image data
  {
    printf("+----------------------------+\n");
    fread(&readInt, sizeof(int), 4, f_trp); readInt = 0;
  	for (int i = 0; i < TRAINING; i++) {
  		for (int j = 0; j < ROWS; j++) {
  			for (int k = 0; k < COLS; k++) {
  				fread(&readInt, 1, 1, f_trp); TRP[i][j][k] = readInt;
  			}
  		}
  		if (i%interval == 0) printf(" Training : Read %5ld images|\r", i);
  	}
    printf("|Training : Read %5ld images|\n", TRAINING);

    fread(&readInt, sizeof(int), 4, f_tep); readInt = 0;
    for (int i = 0; i < TESTING; i++) {
      for (int j = 0; j < ROWS; j++) {
        for (int k = 0; k < COLS; k++) {
          fread(&readInt, 1, 1, f_tep); TEP[i][j][k] = readInt;
        }
      }
      if (i%interval == 0) printf("|Training : Read %5ld images|\r", i);
    }
    printf("|Testing  : Read %5ld images|\n", TESTING);

    printf("+----------------------------+\n\n");
  }

  // Preprocess Data
  {
    printf("+----------------------------+\n");
    printf(" Normalizing Data... \r");
    double average = 0; double sd = 0;
    for (size_t j = 0; j < ROWS; j++) {
      for (size_t k = 0; k < COLS; k++) {
        average = 0; sd = 0;
        for (size_t i = 0; i < TRAINING; i++) {
          average += TRP[i][j][k];
        }
        average /= TRAINING;
        for (size_t i = 0; i < TRAINING; i++) {
          sd += (TRP[i][j][k] - average) * (TRP[i][j][k] - average);
        }
        sd /= (TRAINING - 1);
        sd = sqrt(sd);
        for (size_t i = 0; i < TRAINING; i++) {
          if(sd == 0) TRP[i][j][k] = 0;
          else TRP[i][j][k] = (TRP[i][j][k] - average) / sd;
        }
        TRA[j][k] = average;
        TRD[j][k] = sd;
      }
    }
    printf("|      Normalized  Data      |     \n");
    printf("+----------------------------+\n\n");
  }

  // Init Discriminator Network
  {
    printf("+----------------------------+\n");
    printf(" Init Discriminator...     \r");
    for (size_t i = 0; i < DIN; i++) {
      for (size_t j = 0; j < DHIDDEN; j++) {
        D1W[i][j] = XavierRand(DIN);
      }
    }
    for (size_t i = 0; i < DHIDDEN; i++) {
      D1B[i] = XavierRand(DIN);
    }
    for (size_t i = 0; i < DHIDDEN; i++) {
      for (size_t j = 0; j < DOUT; j++) {
        D2W[i][j] = XavierRand(DHIDDEN);
      }
    }
    for (size_t i = 0; i < DOUT; i++) {
      D2B[i] = XavierRand(DHIDDEN);
    }
    printf("|     Init Discriminator     |     \n");
    printf("+----------------------------+\n\n");
  }

  // Init Generator
  {
    printf("+----------------------------+\n");
    printf(" Init Generator...     \r");
    for (size_t i = 0; i < GIN; i++) {
      for (size_t j = 0; j < GHIDDEN; j++) {
        G1W[i][j] = XavierRand(GIN);
      }
    }
    for (size_t i = 0; i < GIN; i++) {
      G1B[i] = XavierRand(GIN);
    }
    for (size_t i = 0; i < GHIDDEN; i++) {
      for (size_t j = 0; j < GOUT; j++) {
        G2W[i][j] = XavierRand(GHIDDEN);
      }
    }
    for (size_t i = 0; i < GOUT; i++) {
      G2B[i] = XavierRand(GHIDDEN);
    }
    printf("|       Init Generator       |     \n");
    printf("+----------------------------+\n\n");
  }

  // Training
  {
    for (size_t EP = 0; EP < EPOCHS; EP++) {
      double correct = 0;
      for (size_t II = 0; II < TRAINING * 2; II++) {

        if(II % 2 == 0) {
          // Z Vector
          {
            for (size_t i = 0; i < GIN; i++) {
              GINDATA[i] = NormalRand(0,0.5);
            }
          }

          // Generator
          {
            for (size_t i = 0; i < GHIDDEN; i++) {
              GHIDDENDATA[i] = 0;
              for (size_t j = 0; j < GIN; j++) {
                GHIDDENDATA[i] += GINDATA[j] * G1W[j][i];
              }
              GHIDDENDATA[i] += G1B[i];
              GHIDDENDATA[i] = Sigmoid(GHIDDENDATA[i]);
            }

            for (size_t i = 0; i < GOUT; i++) {
              GOUTDATA[i] = 0;
              for (size_t j = 0; j < GHIDDEN; j++) {
                GOUTDATA[i] += GHIDDENDATA[j] * G2W[j][i];
              }
              GOUTDATA[i] += G2B[i];
              GOUTDATA[i] = Sigmoid(GOUTDATA[i]);
            }
          }
          if(II % 10000 == 0){
            for (size_t i = 0; i < ROWS; i++) {
              for (size_t j = 0; j < COLS; j++) {
                printf("%3d ", (int)(GOUTDATA[i*28 + j]*255));
              }
              printf("\n");
            }
            printf("\n");
          }

        }
        // Assignment
        // Fake Image || Real Image
        {
          if(II % 2 == 0) {
            for (size_t i = 0; i < DIN; i++) {
              DINDATA[i] = GOUTDATA[i];
            }
            ANSWER[0] = 0;
            ANSWER[1] = 1;
          } else {
            for (size_t i = 0; i < DIN; i++) {
              DINDATA[i] = TRP[(int)(II-1)/2][i/28][i%28];
            }
            ANSWER[0] = 1;
            ANSWER[1] = 0;
          }
        }

        // Discriminator
        {
          for (size_t i = 0; i < DHIDDEN; i++) {
            DHIDDENDATA[i] = 0;
            for (size_t j = 0; j < DIN; j++) {
              DHIDDENDATA[i] += DINDATA[j] * D1W[j][i];
            }
            DHIDDENDATA[i] += D1B[i];
            DHIDDENDATA[i] = Sigmoid(DHIDDENDATA[i]);
          }

          for (size_t i = 0; i < DOUT; i++) {
            DOUTDATA[i] = 0;
            for (size_t j = 0; j < DHIDDEN; j++) {
              DOUTDATA[i] += DHIDDENDATA[j] * D2W[j][i];
            }
            DOUTDATA[i] += D2B[i];
            DOUTDATA[i] = Sigmoid(DOUTDATA[i]);
          }
        }

        // Error
        {
          for (size_t i = 0; i < DOUT; i++) {
            OUTPUT[i] = DOUTDATA[i];
            // printf("%f ", OUTPUT[i]);
          }
          // printf("\n");

          // Get/Check Answer
          double highest = OUTPUT[0];
          int highestIndex = 0;
          for (size_t i = 0; i < DOUT; i++) {
            if(highest < OUTPUT[i]){
              highest = OUTPUT[i];
              highestIndex = i;
            }
          }
          if(ANSWER[highestIndex]) correct++;

          for (size_t i = 0; i < DOUT; i++) {
            ERROR[i] = OUTPUT[i] - ANSWER[i];
          }

          printf("%d, %f\r", II, correct/II);
        }

        // Discriminator Grad
        {
          // D2B
          for (size_t i = 0; i < DOUT; i++) {
            if(((II / INTERVALS) % 2) == 0)
              OUTERROR[i] = DL * ERROR[i] * (OUTPUT[i] * (1.0 - OUTPUT[i]));
            else
              OUTERROR[i] = GL * ERROR[i] * (OUTPUT[i] * (1.0 - OUTPUT[i]));

            D2BGRAD[i] = OUTERROR[i];
          }

          // D2W
          for (size_t i = 0; i < DOUT; i++) {
            for (size_t j = 0; j < DHIDDEN; j++) {
              D2WGRAD[j][i] = OUTERROR[i] * DHIDDENDATA[j];
            }
          }

          // D1B
          for (size_t i = 0; i < DHIDDEN; i++) {
            double tempErrorSum = 0.0;
            for (size_t j = 0; j < DOUT; j++)
              tempErrorSum += OUTERROR[j] * D2W[i][j];
            HIDDENERROR[i] = tempErrorSum * (DHIDDENDATA[i] * (1.0 - DHIDDENDATA[i]));
            D1BGRAD[i] = HIDDENERROR[i];
          }

          // D1W
          for (size_t i = 0; i < GIN; i++) {
            for (size_t j = 0; j < DHIDDEN; j++) {
              D1WGRAD[i][j] =  HIDDENERROR[j] * DINDATA[i];
            }
          }
        }

        // Generator Grads
        {
          // G2B
          for (size_t i = 0; i < GOUT; i++) {
            double tempErrorSum = 0.0;
            for (size_t j = 0; j < DHIDDEN; j++)
              tempErrorSum += HIDDENERROR[j] * D1W[i][j];
            INPUTERROR[i] = tempErrorSum * (DINDATA[i] * (1.0 - DINDATA[i]));
            G2BGRAD[i] = INPUTERROR[i];
          }

          // G2W
          for (size_t i = 0; i < GHIDDEN; i++) {
            for (size_t j = 0; j < GOUT; j++) {
              G2WGRAD[i][j] =  INPUTERROR[j] * GHIDDENDATA[i];
            }
          }

          // G1B
          for (size_t i = 0; i < GHIDDEN; i++) {
            double tempErrorSum = 0.0;
            for (size_t j = 0; j < GOUT; j++)
              tempErrorSum += INPUTERROR[j] * G2W[i][j];
            GHIDDENERROR[i] = tempErrorSum * (GHIDDENDATA[i] * (1.0 - GHIDDENDATA[i]));
            G1BGRAD[i] = GHIDDENERROR[i];
          }

          // G1W
          for (size_t i = 0; i < GIN; i++) {
            for (size_t j = 0; j < GHIDDEN; j++) {
              G1WGRAD[i][j] = GHIDDENERROR[j] * GINDATA[i];
            }
          }

        }

        // Discriminator Updates
        {
          if(((II / INTERVALS) % 2) == 0) {
            // D2B
            for (size_t i = 0; i < DOUT; i++) {
              D2B[i] -= D2BGRAD[i];
              D2BGRAD[i] = 0;
            }

            // D2W
            for (size_t i = 0; i < DOUT; i++) {
              for (size_t j = 0; j < DHIDDEN; j++) {
                D2W[j][i] -= D2WGRAD[j][i];
                D2WGRAD[j][i] = 0;
              }
            }

            // D1B
            for (size_t i = 0; i < DHIDDEN; i++) {
              D1B[i] -= D1BGRAD[i];
              D1BGRAD[i] = 0;
            }

            // D1W
            for (size_t i = 0; i < GIN; i++) {
              for (size_t j = 0; j < DHIDDEN; j++) {
                D1W[i][j] -= D1WGRAD[i][j];
                D1WGRAD[i][j] = 0;
              }
            }
          }
        }

        // Generator Updates
        {
          if(((II / INTERVALS) % 2) == 1 && II % 2 == 0) {
            for (size_t i = 0; i < GOUT; i++) {
              G2B[i] -= G2BGRAD[i];
            }

            // G2W
            for (size_t i = 0; i < GHIDDEN; i++) {
              for (size_t j = 0; j < GOUT; j++) {
                G2W[i][j] -= G2WGRAD[i][j];
              }
            }

            for (size_t i = 0; i < GHIDDEN; i++) {
              G1B[i] -= G1BGRAD[i];
            }

            // G2W
            for (size_t i = 0; i < GIN; i++) {
              for (size_t j = 0; j < GHIDDEN; j++) {
                G1W[i][j] -= G1WGRAD[i][j];
              }
            }
          }
        }
      }
    }
  }

  return 0;
}

double XavierRand(double fan_in) {
  double int_max = INT_MAX; double sum = 0;
  for (size_t i = 0; i < 12; i++) sum += (float) rand() / RAND_MAX;
  sum -= 6 + 0; sum *= 1 / sqrt(fan_in);
  return sum;
}

double NormalRand(double m, double sd) {
  double int_max = INT_MAX; double sum = 0;
  for (size_t i = 0; i < 12; i++) sum += (float) rand() / RAND_MAX;
  sum -= 6 + m; sum *= 1 / sqrt(sd);
  return sum;
}

double Sigmoid(double x) {
  return (1 / (1 + exp(-1 * x)));
}
