#ifndef _RELU_H_
#define _RELU_H_

void relu(unsigned int oNin,unsigned int oD,float* outputs);


void reluDense(unsigned int num,float* out);

void reluCnn( float* outputs,unsigned int fn, unsigned int inN,unsigned int oD);


void reluDen(float* outputs,unsigned int inN,unsigned int nodeNum);

#endif
