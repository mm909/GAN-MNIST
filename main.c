// ---------- //
// Mikian Musser
// ---------- //

#include <stdio.h>

// MNIST Data
#define TRAINING 60000
#define TESTING  10000
#define ROWS        28
#define COLS        28
#define TRIF "data/train-images.idx3-ubyte"
#define TRLF "data/train-labels.idx1-ubyte"
#define TEIF  "data/t10k-images.idx3-ubyte"
#define TELF  "data/t10k-labels.idx1-ubyte"

// Image Data
double TP[TRAINING + TESTING][ROWS][COLS];
double TL[TRAINING + TESTING];

int main() {

  FILE * f_trp = fopen(TRIF, "rb");
  FILE * f_trl = fopen(TRLF, "rb");
  FILE * f_tep = fopen(TEIF, "rb");
  FILE * f_tel = fopen(TELF, "rb");
  if (!f_trp || !f_trl || !f_tep || !f_tel) printf("Error Opening Data Files.");

  int readInt = 0;
  fread(&readInt, sizeof(int), 4, f_trp);
  readInt = 0;
  

  int interval = TRAINING / 100;
	for (int i = 0; i < TRAINING; i++) {
		for (int j = 0; j < ROWS; j++) {
			for (int k = 0; k < COLS; k++) {
				fread(&readInt, 1, 1, f_trp);
				TP[i][j][k] = readInt;
			}
		}
		if (i%interval == 0) printf("Training     : Read %5ld images\r", i);
	}

  return 0;
}
