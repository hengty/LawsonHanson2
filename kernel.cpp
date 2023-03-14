//lawson-hanson NNLS machine -- kernel
//ty@wisc.edu
//Last update: 10 Mar 2023

#include "kernel_util.hpp"

//A * x = ydata, produce x
extern "C" void lawson_hanson(float ydata_buff[WMAX], float ALUT_buff[ALUTlenMAX], int ydata_len, int ALUTlen, int bpt, int offset,
		                      float xpulses_buff[XMAX], int iterMAX=ITERMAX, float tolerance=0.001)
{
	float xsol[XMAX] = {0.0};   //solution initialized to all zero
	float ydata[WMAX], QTydata[WMAX];
	float y_Ax[WMAX];   //for storing y-Ax
	float lls[WMAX];   //for storing the ordinary linear least square sub result in each iteration
	bool px[XMAX] = {false};   //indices of current basis SPE column vectors in consideration
	int xsol_len = bpt*ydata_len;   //this cannot be larger than XMAX
	int ipx2iog[WMAX];   //list of px indices true -- for converting px indices back to the original indices of xsol
	LUT ALUT(ALUTlen, bpt, offset, ALUT_buff);   //emulate the A matrix with a much smaller lookup table
	ydata_import: for(int i=0; i<ydata_len; i++) {ydata[i]=ydata_buff[i], QTydata[i]=ydata_buff[i];}

	int gcounter=0, Glistrow[GMAX];
	float Glist[GMAX][2], Rmat[WMAX][XMAX];   //lls via QR factorization via Givens rotations

	//Lawson Hanson NNLS
	bool px_ready;
	int inext, iter=0, sumpx=1, ip;
	int numdel, delcol[DELCOLMAX];   //is DELCOLMAX=10 large enough?
	float alpha, alpha_min;   //ngradmax is some sort of projected error measuring agreement between solution and data
	float ngradmax = turn_on_next_p(ALUT, ydata, px, xsol_len, ydata_len, &inext);   //note: this turns on the px's index of ngradmax
	QRaddcol(ALUT, Rmat, Glist, QTydata, Glistrow, ipx2iog, inext, ydata_len, sumpx, &gcounter);   //update the QR factorization, and also ipx2iog
	thewhile_loop:
	while(ngradmax>tolerance && iter<iterMAX && sumpx<=ydata_len)
	{
		//update the ordinary linear least square result to reflect the changed QR
		lls_init: for(int i=0; i<sumpx; i++)
			#pragma HLS loop_tripcount min=1 max=WMAX
			lls[i] = QTydata[i];   //start off as Qt*ydata
		lls_QR(lls, Rmat, ipx2iog, sumpx);

		//check for any nonpositive lls[i], determining alpha_min
		alpha_min = 1.0;   //note: alpha can never be greater than 1.0
		compute_alpha:
		for(int i=0; i<sumpx; i++)
			#pragma HLS LOOP_TRIPCOUNT max=WMAX
			if(lls[i]<ZEROTHRES){
				alpha = xsol[ipx2iog[i]]/(xsol[ipx2iog[i]]-lls[i]);
				if(alpha < alpha_min) alpha_min = alpha;}

		//update the active part of xsol, and then turn off any resulting p vector that is no longer positive
		numdel=0, px_ready = true;
		update_xsol:
		for(int i=0; i<sumpx; i++)
		{
			#pragma HLS dependence variable=xsol type=inter false
			#pragma HLS loop_tripcount max=WMAX
			ip = ipx2iog[i];
			xsol[ip] += alpha_min*(lls[i]-xsol[ip]);
			if(xsol[ip]<ZEROTHRES){
				px_ready = false;   //not ready to turn on a new p vector if have to turn at least one off
				delcol[numdel] = ip;
				numdel+=1;}
		}

		//turn on a new p vector, the one with the new max ngrad, if no p vector was turned off from previous iteration
		if(px_ready)
		{
			compute_y_Ax:
			for(int i=0; i<ydata_len; i++)
			{
				#pragma HLS loop_tripcount max=WMAX
				y_Ax[i] = 0.0;
				for(int j=0; j<sumpx; j++)
					#pragma HLS loop_tripcount max=WMAX
					y_Ax[i] -= ALUT.eval(i,ipx2iog[j]) * xsol[ipx2iog[j]];
				y_Ax[i] += ydata[i];// - ALUT.evalprodx_i(i, xsol, sumpx, ipx2iog);
			}

			ngradmax = turn_on_next_p(ALUT, y_Ax, px, xsol_len, ydata_len, &inext);   //note: this turns on the px's index of ngradmax
			sumpx+=1;
			QRaddcol(ALUT, Rmat, Glist, QTydata, Glistrow, ipx2iog, inext, ydata_len, sumpx, &gcounter);
		}
		else
			delcol:
			for(int i=0; i<numdel; i++){
				#pragma HLS loop_tripcount max=1
				px[delcol[i]] = false;
				sumpx-=1;
				QRdelcol(Rmat, Glist, QTydata, Glistrow, ipx2iog, delcol[i], sumpx, &gcounter);}   //this also updates ipx2iog

		printf("iter: %d, sumpx: %d, ngradmax: %f, alpha_min: %f, px_ready: %d, inext: %d, gcounter: %d   [", iter, sumpx, ngradmax, alpha_min, px_ready, inext, gcounter);
		for(int j=0; j<sumpx; j++) if(ipx2iog[j]!=inext) printf("%d, ", ipx2iog[j]);
		printf("]   [");
		for(int j=0; j<sumpx; j++) if(ipx2iog[j]!=inext) printf("%f, ", xsol[ipx2iog[j]]);
		printf("]\n");

		iter++;
	}

	//copy final result to the buffer to be transfer back to host
	send_result: for(int i=0; i<xsol_len; i++) xpulses_buff[i] = xsol[i];

}
