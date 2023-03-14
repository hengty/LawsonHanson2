//lawson-hanson NNLS machine -- kernel utility function definitions. Apparently, can't put function definitions in .hpp file...
//ty@wisc.edu
//Last update: 10 Mar 2023

#include "kernel_util.hpp"


//compute gradient and identify which non p vector produces the largest negative grad
float turn_on_next_p(LUT ALUT, float y_Ax[WMAX], bool px[XMAX], int xsol_len, int ydata_len, int* inext)
{
	float ngradi, ngradmax=-1.0;
	int imax;
	find_nextp:
	for(int i=0; i<xsol_len; i++)
		#pragma HLS loop_tripcount max=XMAX
		if(!px[i])
		{
			ngradi=0.0;   //ALUT.evalprody_j(i, y_Ax, ydata_len);   //At(y-Ax)
			for(int j=0; j<ydata_len; j++)
				#pragma HLS loop_tripcount max=WMAX
				ngradi += ALUT.eval(j,i)*y_Ax[j];
			if(ngradi>ngradmax){imax=i, ngradmax=ngradi;}
		}

	px[imax] = true;
	*inext = imax;
	return ngradmax;
}


//QR factorization with the Givens rotations method. (Could be more efficient than Householder for sparse matrices)
//"Updating the QR factorization and the least squares problem" -- Hammarling2008
//http://eprints.ma.man.ac.uk/1192/1/qrupdating_12nov08.pdf
//the cases for calculating the Givens matrix is to avoid numerical instability
void Givens(float bot, float top, float* cosa, float* sina)
{
	//Elements of the Givens matrix: cosa, sina
	float t;
    if(bot==0.0){*cosa = 1.0; *sina = 0.0;}
    else if(std::abs(bot)>=std::abs(top)){
        t = -top/bot;
        *sina = 1/sqrt(1+t*t);
        *cosa = *sina*t;}
    else{
        t = -bot/top;
        *cosa = 1/sqrt(1+t*t);
        *sina = *cosa*t;}
}

void QRaddcol(LUT ALUT, float Rmat[WMAX][XMAX], float Glist[GMAX][2], float QTydata[WMAX], int Glistrow[GMAX], int ipx2iog[WMAX],
		int newcol, int ydata_len, int size, int* gcounter)
{
	bool found=false, started=false;
	int inewcol, bot, top, jp, rowbot;
	float cosa, sina;   //[cosa, sina] of the Givens matrix
	float temp, temp2;   //for storing temporary values

	//find inewcol's spot in ipx2iog, update ipx2iog
	ipx2iog_update:
	for(int i=size-1; i>-1; i--)
		#pragma HLS LOOP_TRIPCOUNT max=WMAX
		if(!found){
			if(ipx2iog[i-1]<newcol || i==0){
				found = true;
				ipx2iog[i] = newcol;
				inewcol = i;}   //the column position in the submatrix to add. Also equals to number of rows to zero out
			else ipx2iog[i] = ipx2iog[i-1];}

	//copy the new column to Rmat
	newcol_copy:
	for(int j=0; j<ydata_len; j++)
		#pragma HLS LOOP_TRIPCOUNT max=WMAX
		Rmat[j][newcol] = ALUT.eval(j, newcol);   //the newly added column starts off a copy from A
	newcol_init:
	for(int j=0; j<*gcounter; j++){   //then applied with Qtranpose
		#pragma HLS LOOP_TRIPCOUNT max=GMAX
		cosa=Glist[j][0], sina=Glist[j][1], bot=Glistrow[j];
		top=bot-1;
		temp = cosa*Rmat[top][newcol] - sina*Rmat[bot][newcol];
		Rmat[bot][newcol] = sina*Rmat[top][newcol] + cosa*Rmat[bot][newcol];
		Rmat[top][newcol] = temp;}

    //Givens rotations
	givens1:
    for(int i=ydata_len-1; i>inewcol; i--)
    {
		#pragma HLS LOOP_TRIPCOUNT max=WMAX
    	if(!started && std::abs(Rmat[i][newcol])>=ZEROTHRES) started=true;
    	if(started)
    	{
    		bot = i, top = i-1;
    		Givens(Rmat[bot][newcol], Rmat[top][newcol], &cosa, &sina);
    		Rmat[top][newcol] = cosa*Rmat[top][newcol] - sina*Rmat[bot][newcol];
    		Rmat[bot][newcol] = 0.0;
    		givens2:
    		for(int j=bot; j<size; j++){   //apply Givens to the rest of the necessary columns
				#pragma HLS LOOP_TRIPCOUNT max=WMAX
            	jp = ipx2iog[j];
            	temp = cosa*Rmat[top][jp] - sina*Rmat[bot][jp];
            	Rmat[bot][jp] = sina*Rmat[top][jp] + cosa*Rmat[bot][jp];
            	Rmat[top][jp] = temp;}

    		//update QT*ydata
    		temp2 = cosa*QTydata[top] - sina*QTydata[bot];
    		QTydata[bot] = sina*QTydata[top] + cosa*QTydata[bot];
    		QTydata[top] = temp2;

    		//record the Givens rotation
    		Glist[*gcounter][0] = cosa;
    		Glist[*gcounter][1] = sina;
    		Glistrow[*gcounter] = bot;
			*gcounter+=1;
    	}
    }
}


void QRdelcol(float Rmat[WMAX][XMAX], float Glist[GMAX][2], float QTydata[WMAX], int Glistrow[GMAX], int ipx2iog[WMAX],
		int delcol, int size, int* gcounter)
{
	bool found=false;
	int idelcol, bot, top, jp;
	float cosa, sina;   //transpose of the Givens matrix
	float temp, temp2;

	//update ipx2iog, deleting delcol
	ipx2iog_update:
	for(int i=0; i<size; i++){
		#pragma HLS LOOP_TRIPCOUNT max=(WMAX-1)
    	if(ipx2iog[i]==delcol){
        	found = true;
			idelcol = i;}
    	if(found) ipx2iog[i] = ipx2iog[i+1];}

	//Given rotations
	givens1:
	for(int i=idelcol; i<size; i++)
	{
		#pragma HLS LOOP_TRIPCOUNT max=WMAX
		bot = i+1, top = i, jp = ipx2iog[i];
		Givens(Rmat[bot][jp], Rmat[i][jp], &cosa, &sina);

		//update Rmat
		Rmat[top][jp] = cosa*Rmat[top][jp] - sina*Rmat[bot][jp];
		Rmat[bot][jp] = 0.0;
		Rmat_update:
		for(int j=bot; j<size; j++){
			#pragma HLS LOOP_TRIPCOUNT max=WMAX
			jp = ipx2iog[j];
			temp = cosa*Rmat[top][jp] - sina*Rmat[bot][jp];
			Rmat[bot][jp] = sina*Rmat[top][jp] + cosa*Rmat[bot][jp];
			Rmat[top][jp] = temp;}

		//update QT*ydata
		temp2 = cosa*QTydata[top] - sina*QTydata[bot];
		QTydata[bot] = sina*QTydata[top] + cosa*QTydata[bot];
		QTydata[top] = temp2;

		//record the Givens rotation
		Glist[*gcounter][0] = cosa;
		Glist[*gcounter][1] = sina;
		Glistrow[*gcounter] = bot;
		*gcounter+=1;
	}

}


void lls_QR(float lls[WMAX], float Rmat[WMAX][XMAX], int ipx2iog[WMAX], int size)
{
	//lls computation by back substitution in R*lls=Qt*ydata
	lls_backsub:
	for(int i=size-1; i>-1; i--)   //back substition to solve for lls in R*lls=Qt*ydata
	{
		#pragma HLS loop_tripcount min=1 max=WMAX
		for(int j=i+1; j<size; j++)
			#pragma HLS loop_tripcount min=1 max=WMAX
			lls[i] -= lls[j]*Rmat[i][ipx2iog[j]];
		lls[i] /= Rmat[i][ipx2iog[i]];
	}
}













