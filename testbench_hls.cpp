//testbench to debug lawson_hanson kernel. Why is it stalling in hardware emulation?
//ty@wisc.edu
//last update: 10 Mar 2023

#include <iostream>
#include "testbench_util.hpp"
#include "kernel_util.hpp"

int main()
{
	ALUTclass ALUT(12, 4, 0.375);   //ilen=12, bpt_in=4, bpt_offset_in=0.375
	const int ydata_len = 128;//30;//
	const int xdata_len = ydata_len*ALUT.bpt;
	float xpulses_buff[xdata_len];
	float ydata_buff[ydata_len] = {1,2,1,1,1,1,1,3,18,113,268,360,436,459,426,377,315,244,183,145,
			                       121,97,83,74,64,57,56,57,57,58,
			                       64,64,58,52,49,45,41,41,36,27,
	                               23,21,18,17,21,22,21,20,19,15,14,16,15,12,11,10,6,5,8,9,
                                   9,9,10,11,10,7,6,8,8,5,5,5,6,6,4,5,5,3,2,2,
								   1,1,2,4,3,3,3,3,3,3,3,3,2,1,1,2,3,3,2,2,
								   2,2,2,3,3,3,2,0,1,1,2,1,1,0,1,0,0,1,0,2,2,3,3,2,1,0,1,1};

	//DUT
	lawson_hanson(ydata_buff, ALUT.ALUT, ydata_len, ALUT.ALUTlen, ALUT.bpt, ALUT.offset, xpulses_buff, 1000, 0.001);

	//print result
	std::cout<<"\nTestbench print:\n[";
	for(int i=0; i<xdata_len; i++) std::cout<<xpulses_buff[i]<<",";
	std::cout<<"]\n";

	return 0;
}
