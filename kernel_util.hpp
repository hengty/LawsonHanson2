//lawson-hanson NNLS machine -- kernel utilities functions
//ty@wisc.edu
//Last update: 10 Mar 2023

#define WMAX 384   //128ATWD+256FADC
#define XMAX 4*WMAX   //4 basis spe per time bin
#define ALUTlenMAX 128
#define ITERMAX 1000   //maximum number of Lawson Hanson iterations
#define ZEROTHRES 1e-8   //this is sort of arbitrary
#define GMAX WMAX*WMAX   //length of list storing Givens rotations. Not sure if this is long enough
#define DELCOLMAX 10   //max number of column vectors that can be removed from active set per iteration. Hopefully 10 is enough.

#include <iostream>   //printf and cout for debugging

#include "ap_fixed.h"
#include <stdint.h>
#include "hls_math.h"

struct LUT{
	int len, bpt, offset;
	float lut[ALUTlenMAX];
	LUT(int len_in, int bpt_in, int offset_in, float lut_in[ALUTlenMAX])
	{
		len=len_in, bpt=bpt_in, offset=offset_in;
		LUT_LUT: for(int i=0; i<len; i++) lut[i] = lut_in[i];
	}
	float eval(int i, int j)
	{
		//#pragma HLS array_partition variable=lut
		int LUTarg = bpt*i-j-offset;   //mapping the matrix indices pair ij to lut
		return (-1<LUTarg && LUTarg<len) ? lut[LUTarg] : 0.0;
	}/*
	float evalprodx_i(int i, float x[XMAX], int sumpx, int ipx2iog[WMAX])   //compute Ax[i]
	{
		float Ax_i=0.0;
		LUT_evalprodx_i:
		for(int j=0; j<sumpx; j++)
			#pragma HLS loop_tripcount max=WMAX
			Ax_i += eval(i,ipx2iog[j]) * x[ipx2iog[j]];
		return Ax_i;
	}
	float evalprody_j(int j, float y[WMAX], int y_len)   //compute the jth component of Atranspose y
	{
		float Ay_j=0.0;
		LUT_evalprody_j: for(int i=0; i<WMAX; i++) if(i<y_len) Ay_j += eval(i,j)*y[i];
		return Ay_j;
	}*/
};

extern "C" void lawson_hanson(float ydata_buff[WMAX], float ALUT_buff[ALUTlenMAX], int ydata_len, int ALUTlen, int bpt, int offset,
		                      float xpulses_buff[XMAX], int iterMax, float tolerance);

float turn_on_next_p(LUT ALUT, float y_Ax[WMAX], bool px[XMAX], int xsol_len, int ydata_len, int* inext);

void QRaddcol(LUT ALUT, float Rmat[WMAX][XMAX], float Glist[GMAX][2], float QTydata[WMAX], int Glistrow[GMAX], int ipx2iog[WMAX],
		int newcol, int ydata_len, int size, int* gcounter);

void QRdelcol(float Rmat[WMAX][XMAX], float Glist[GMAX][2], float QTydata[WMAX], int Glistrow[GMAX], int ipx2iog[WMAX],
		int delcol, int size, int* gcounter);

void lls_QR(float lls[WMAX], float Rmat[WMAX][XMAX], int ipx2iog[WMAX], int size);





