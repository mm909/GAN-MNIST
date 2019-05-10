// ---------- //
// Mikian Musser
// ---------- //

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <time.h>
#include <math.h>

// Macro Vars
#define EPOCHS 2

// Discriminator Vars
#define DL          0.1
#define DFRAMES       6
#define DWINDOWSIZE   5
#define DHIDDEN      40

// Discriminator Constants
#define DOUT                           2
#define DF1SIZE  (ROWS - DWINDOWSIZE + 1)
#define DF2SIZE             (DF1SIZE / 2)
#define DPOOLSIZE                      2

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

// Discriminator Trainable Paramaters
double CLW[DFRAMES][DWINDOWSIZE][DWINDOWSIZE];   // Convolutional Layer Weights
double CLB[DFRAMES];                           // Convolutional Layer Biases
double FCL1B[DHIDDEN];                         // Fully Connected Layer 1 Biases
double FCL1W[DHIDDEN][DFRAMES][DF2SIZE][DF2SIZE]; // Fully Connected Layer 1 Weights
double FCL2B[DOUT];                            // Fully Connected Layer 2 Biases
double FCL2W[DOUT][DHIDDEN];                    // Fully Connected Layer 2 Weights


// Generator Trainable Paramaters
double GENFCL1W[100][512];
double GENFCL1B[512];
double GENFCL2W[512][784];
double GENFCL2B[784];

// Discriminator Grads
double CLWGRAD[DFRAMES][DWINDOWSIZE][DWINDOWSIZE];   // Convolutional Layer Weights
double CLBGRAD[DFRAMES];                           // Convolutional Layer Biases
double FCL1BGRAD[DHIDDEN];                         // Fully Connected Layer 1 Biases
double FCL1WGRAD[DHIDDEN][DFRAMES][DF2SIZE][DF2SIZE]; // Fully Connected Layer 1 Weights
double FCL2BGRAD[DOUT];                            // Fully Connected Layer 2 Biases
double FCL2WGRAD[DOUT][DHIDDEN];

// Discriminator data
double INPUT[ROWS][COLS];
double CLS[DFRAMES][DF1SIZE][DF1SIZE];   // Convolutional Layer S Value (Before Activation)
double CLO[DFRAMES][DF1SIZE][DF1SIZE];   // Convolutional Layer O Value (After Activation)
double PLT[DFRAMES][DF2SIZE][DF2SIZE];   // Pooling Layer "Pixel Tracker" (0-3) ((2*j) + i)
double PL[DFRAMES][DF2SIZE][DF2SIZE];    // Pooling Layer Values
double FCL1S[DHIDDEN];                 // Fully Connected Layer 1 S Values
double FCL1O[DHIDDEN];                 // Fully Connected Layer 1 O Values
double FCL2S[DOUT];                    // Fully Connected Layer 2 S Values
double FCL2O[DOUT];                    // Fully Connected Layer 2 O Values
double OUTPUT[DOUT];                   // Ouput Values of feedforrward
double ANSWER[DOUT];                   // Answer array for training image

// Generator Data
double GENIN[100];
double GENHIDDEN[512];
double GENOUT[784];

double ERROR[DOUT];                   // Answer array for training image
double OUTPUTERROR[DOUT];                // Temp Value for Output Back prop
double HIDDENSUMERROR[DHIDDEN];          // Temp Value for Hidden Back prop
double FIRSTLAYERSUMERROR[DFRAMES];      // Temp Value for First layer back prop
double CLERROR[DFRAMES][DF2SIZE][DF2SIZE]; // Temp Value for First layer back prop

double XavierRand(double fan_in);
double sigmoid(double x);
double RandNormal(double m, double sd);

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
    printf("+---------------+\n");
    printf(" Normalizing Data... \r");
    for (size_t j = 0; j < ROWS; j++) {
      for (size_t k = 0; k < COLS; k++) {
        double average = 0; double sd = 0;
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
        average = 0; sd = 0;
        for (size_t i = 0; i < TESTING; i++) {
          average += TEP[i][j][k];
        }
        average /= TESTING;
        for (size_t i = 0; i < TESTING; i++) {
          sd += (TEP[i][j][k] - average) * (TEP[i][j][k] - average);
        }
        sd /= (TESTING - 1);
        sd = sqrt(sd);
        for (size_t i = 0; i < TESTING; i++) {
          if(sd == 0) TEP[i][j][k] = 0;
          else TEP[i][j][k] = (TEP[i][j][k] - average) / sd;
        }
      }
    }
    printf("|Normalized Data|     \n");
    printf("+---------------+\n\n");
  }

  // Init Discriminator Network
  {
    printf("+------------------+\n");
    printf(" Init Discriminator...     \r");
    for (size_t i = 0; i < DFRAMES; i++) {
      for (size_t j = 0; j < DWINDOWSIZE; j++) {
        for (size_t k = 0; k < DWINDOWSIZE; k++) {
            CLW[i][j][k] = XavierRand(DWINDOWSIZE * DWINDOWSIZE);
        }
      }
    }
    for (size_t i = 0; i < DFRAMES; i++) {
      CLB[i] = XavierRand(DWINDOWSIZE * DWINDOWSIZE);
    }
    for (size_t i = 0; i < DHIDDEN; i++) {
      for (size_t j = 0; j < DFRAMES; j++) {
        for (size_t k = 0; k < DF2SIZE; k++) {
          for (size_t l = 0; l < DF2SIZE; l++) {
            FCL1W[i][j][k][l] = XavierRand(DF2SIZE * DF2SIZE * DFRAMES);
          }
        }
      }
    }
    for (size_t i = 0; i < DHIDDEN; i++) {
      FCL1B[i] = XavierRand(DF2SIZE * DF2SIZE * DFRAMES);
    }
    for (size_t i = 0; i < DOUT; i++) {
      for (size_t j = 0; j < DHIDDEN; j++) {
        FCL2W[i][j] = XavierRand(DHIDDEN);
      }
    }
    for (size_t i = 0; i < DOUT; i++) {
      FCL2B[i] = XavierRand(DHIDDEN);
    }
    printf("|Init Discriminator|     \n");
    printf("+------------------+\n\n");
  }

  // Init Generator
  {
    printf("+--------------+\n");
    printf(" Init Generator...     \r");
    for (size_t i = 0; i < 100; i++) {
      for (size_t j = 0; j < 512; j++) {
        GENFCL1W[i][j] = XavierRand(100);
      }
    }
    for (size_t i = 0; i < 512; i++) {
      GENFCL1B[i] = XavierRand(100);
    }
    for (size_t i = 0; i < 512; i++) {
      for (size_t j = 0; j < 784; j++) {
        GENFCL2W[i][j] = XavierRand(512);
      }
    }
    for (size_t i = 0; i < 784; i++) {
      GENFCL2B[i] = XavierRand(512);
    }
    printf("|Init Generator|     \n");
    printf("+--------------+\n\n");
  }

  // Start Training
  {
    double correctCount = 0;
    double averageError = 0;
    clock_t t = clock();

    for(size_t EP = 0; EP < EPOCHS; EP++) {
      correctCount = 0;
      averageError = 0;
      for (size_t II = 0; II < TRAINING*2; II++) {

        if(II%2 == 0){
          for (size_t i = 0; i < ROWS; i++) {
            for (size_t j = 0; j < COLS; j++) {
              INPUT[i][j] = TRP[(int)(II/2)][i][j];
            }
          }
        } else {
          // INPUT[i][j] = RandNormal(0,1);
          {
            for (size_t k = 0; k < 100; k++) {
              GENIN[k] = RandNormal(0,1);
            }

            for (size_t k = 0; k < 512; k++) {
              GENHIDDEN[k] = 0;
              for (size_t l = 0; l < 100; l++) {
                GENHIDDEN[k] += GENIN[l] * GENFCL1W[l][k];
              }
              GENHIDDEN[k] += GENFCL1B[k];
              GENHIDDEN[k] = sigmoid(GENHIDDEN[k]);
            }

            for (size_t k = 0; k < 784; k++) {
              GENOUT[k] = 0;
              for (size_t l = 0; l < 512; l++) {
                GENOUT[k] += GENHIDDEN[l] * GENFCL2W[l][k];
              }
              GENOUT[k] += GENFCL2B[k];
              GENOUT[k] = sigmoid(GENOUT[k]);
            }

            for (size_t i = 0; i < ROWS; i++) {
              for (size_t j = 0; j < COLS; j++) {
                if(TRD[i][j] == 0) INPUT[i][j] = 0;
                else INPUT[i][j] = (GENOUT[(i*28)+j] - TRA[i][j]) / TRD[i][j];
              }
            }
          }
        }

        for (size_t i = 0; i < DFRAMES; i++) {
          for (size_t j = 0; j < DF1SIZE; j++) {
            for (size_t k = 0; k < DF1SIZE; k++) {
              CLS[i][j][k] = 0.0;
              for (size_t l = 0; l < DWINDOWSIZE; l++) {
                for (size_t m = 0; m < DWINDOWSIZE; m++) {
                  CLS[i][j][k] += INPUT[j + l][k + m] * CLW[i][l][m];
                }
              }
              CLS[i][j][k] += CLB[i];
              CLO[i][j][k] = sigmoid(CLS[i][j][k]);
            }
          }
        }

        for (size_t i = 0; i < DFRAMES; i++) {
          for (size_t j = 0; j < DF2SIZE; j++) {
            for (size_t k = 0; k < DF2SIZE; k++) {
              double max  = CLO[i][j * 2][k * 2];
              PLT[i][j][k] = 0;
              if(CLO[i][j * 2][(k * 2) + 1] > max) {
                max = CLO[i][j * 2][(k * 2) + 1];
                PLT[i][j][k] = 1;
              }
              if(CLO[i][(j * 2) + 1][k * 2] > max) {
                max = CLO[i][(j * 2) + 1][k * 2];
                PLT[i][j][k] = 2;
              }
              if(CLO[i][(j * 2) + 1][(k * 2) + 1] > max) {
                max = CLO[i][(j * 2) + 1][(k * 2) + 1];
                PLT[i][j][k] = 3;
              }
              PL[i][j][k] = max;
            }
          }
        }

        for (size_t i = 0; i < DHIDDEN; i++) {
          FCL1S[i] = 0;
          for (size_t j = 0; j < DFRAMES; j++) {
            for (size_t k = 0; k < DF2SIZE; k++) {
              for (size_t l = 0; l < DF2SIZE; l++) {
                FCL1S[i] += PL[j][k][l] * FCL1W[i][j][k][l];
              }
            }
          }
          FCL1S[i] += FCL1B[i];
          FCL1O[i] = sigmoid(FCL1S[i]);
        }

        for (size_t i = 0; i < DOUT; i++) {
          FCL2S[i] = 0;
          for (size_t j = 0; j < DHIDDEN; j++) {
            FCL2S[i] += FCL1O[j] * FCL2W[i][j];
          }
          FCL2S[i] += FCL2B[i];
          FCL2O[i] = sigmoid(FCL2S[i]);
        }

        for (size_t i = 0; i < DOUT; i++) {
          OUTPUT[i] = FCL2O[i];
        }

        if(II % 2 == 0){
          ANSWER[0] = 1;
          ANSWER[1] = 0;
        } else {
          ANSWER[0] = 0;
          ANSWER[1] = 1;
        }

        double highest = OUTPUT[0];
        int highestIndex = 0;
        for (size_t i = 0; i < DOUT; i++) {
          if(highest < OUTPUT[i]){
            highest = OUTPUT[i];
            highestIndex = i;
          }
        }
        if(ANSWER[highestIndex]) correctCount++;

        for (size_t i = 0; i < DOUT; i++) {
          ERROR[i] = OUTPUT[i] - ANSWER[i];
          averageError += ERROR[i] * ERROR[i] * 0.5;
        }

        for (size_t i = 0; i < DOUT; i++) {
          OUTPUTERROR[i] = DL * ERROR[i] * (FCL2O[i] * (1.0 - FCL2O[i]));
          FCL2BGRAD[i] += OUTPUTERROR[i];
        }

        for (size_t i = 0; i < DOUT; i++) {
          for (size_t j = 0; j < DHIDDEN; j++) {
            FCL2WGRAD[i][j] += OUTPUTERROR[i] * FCL1O[j];
          }
        }

        for (size_t i = 0; i < DHIDDEN; i++) {
          double errorSum = 0.0;
          for (size_t j = 0; j < DOUT; j++){
            errorSum += OUTPUTERROR[j] * FCL2W[j][i];
          }
          HIDDENSUMERROR[i] = errorSum * (FCL1O[i] * (1.0 - FCL1O[i]));     // Sigmoid
          FCL1BGRAD[i] += HIDDENSUMERROR[i];
        }

        for (size_t i = 0; i < DHIDDEN; i++) {
          for (size_t j = 0; j < DFRAMES; j++) {
            for (size_t k = 0; k < DF2SIZE; k++) {
              for (size_t l = 0; l < DF2SIZE; l++) {
                FCL1WGRAD[i][j][k][l] += HIDDENSUMERROR[i] * PL[j][k][l];
              }
            }
          }
        }

        for (size_t i = 0; i < DFRAMES; i++) {
          FIRSTLAYERSUMERROR[i] = 0.0;
          for (size_t j = 0; j < DF2SIZE; j++) {
            for (size_t k = 0; k < DF2SIZE; k++) {
              double errorSum = 0.0;
              CLERROR[i][j][k] = 0.0;
              for (size_t l = 0; l < DHIDDEN; l++){
                errorSum += HIDDENSUMERROR[l] * FCL1W[l][i][j][k];
              }
              CLERROR[i][j][k] = errorSum * (PL[i][j][k] * (1.0 -  PL[i][j][k]));
              FIRSTLAYERSUMERROR[i] += errorSum * (PL[i][j][k] * (1.0 -  PL[i][j][k]));  // Sigmoid
            }
          }
          CLBGRAD[i] += FIRSTLAYERSUMERROR[i];
        }

        for (size_t i = 0; i < DFRAMES; i++) {
          for (size_t j = 0; j < DWINDOWSIZE; j++) {
            for (size_t k = 0; k < DWINDOWSIZE; k++) {
              double errorSum = 0.0;
              for (size_t m = 0; m < DF2SIZE; m++) {
                for (size_t n = 0; n < DF2SIZE; n++) {
                  int i3 = (int)PLT[i][m][n] / 2;
                  int j3 = (int)PLT[i][m][n] % 2;
                  errorSum += CLERROR[i][m][n] * INPUT[2*m + j + i3][2*n + k + j3];
                }
              }
              CLWGRAD[i][j][k] += errorSum;
            }
          }
        }

        for (size_t i = 0; i < DOUT; i++) {
          FCL2B[i] -= (FCL2BGRAD[i]);
          FCL2BGRAD[i] = 0;
        }

        for (size_t i = 0; i < DOUT; i++) {
          for (size_t j = 0; j < DHIDDEN; j++) {
            FCL2W[i][j] -= (FCL2WGRAD[i][j]);
            FCL2WGRAD[i][j] = 0;
          }
        }

        for (size_t i = 0; i < DHIDDEN; i++) {
          FCL1B[i] -= (FCL1BGRAD[i]);
          FCL1BGRAD[i] = 0;
        }

        for (size_t i = 0; i < DHIDDEN; i++) {
          for (size_t j = 0; j < DFRAMES; j++) {
            for (size_t k = 0; k < DF2SIZE; k++) {
              for (size_t l = 0; l < DF2SIZE; l++) {
                FCL1W[i][j][k][l] -= (FCL1WGRAD[i][j][k][l]);
                FCL1WGRAD[i][j][k][l] = 0;
              }
            }
          }
        }

        for (size_t i = 0; i < DFRAMES; i++) {
          CLB[i] -= (CLBGRAD[i]);
          CLBGRAD[i] = 0;
        }

        for (size_t i = 0; i < DFRAMES; i++) {
          for (size_t l = 0; l < DWINDOWSIZE; l++) {
            for (size_t m = 0; m < DWINDOWSIZE; m++) {
              CLW[i][l][m] -= (CLWGRAD[i][l][m]);
              CLWGRAD[i][l][m] = 0;
            }
          }
        }





















        if(II % 100 == 0 && II != 0){
          printf("Time %.0fs --- Epoch %d ---  Image %ld --- Accuracy %0.5g%% --- Error %0.5g%%           \r", floor(((double)(clock() - t)) / CLOCKS_PER_SEC), EP, II, ((double)correctCount / II) * 100, (averageError / II) * 100);
        }
      } // Training
    } // Epoch
  }

  return 0;
}

double XavierRand(double fan_in) {
  double int_max = INT_MAX; double sum = 0;
  for (size_t i = 0; i < 12; i++) sum += (float) rand() / RAND_MAX;
  sum -= 6 + 0; sum *= 1 / sqrt(fan_in);
  return sum;
}

double RandNormal(double m, double sd) {
  double int_max = INT_MAX; double sum = 0;
  for (size_t i = 0; i < 12; i++) sum += (float) rand() / RAND_MAX;
  sum -= 6 + m; sum *= 1 / sqrt(sd);
  return sum;
}


double sigmoid(double x) {
  return (1 / (1 + exp(-1 * x)));
}
