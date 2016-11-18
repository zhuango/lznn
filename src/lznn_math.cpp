#ifndef _LZNN_MATH_HPP_
#define _LZNN_MATH_HPP_
#include "lznn_types.h"


void xDotApb(Vector &x,
		Matrix &A,
		Vector &b,
		Vector &results)
{
	int row = x.size();
	int col = results.size();
	
	for(int i = 0; i < col; i++)
	{
		results[i] = b[i];
	}
	
	for(int i = 0; i < row; i++)
	{
		for(int j = 0; j < col; j++)
		{
			results[j] += A[i][j] * x[i];
		}
	}
}

void Axpy(Matrix &A, 
		Vector &x,
		Vector &results)
{
	int row = results.size();
	int col = x.size();
	
	for(int i = 0; i < row; i++)
	{
		for(int j = 0; j < col; j++)
		{
			results[i] += A[i][j] * x[j];
		}
	}
}


void Axpb(Matrix &A, 
		Vector &x,
		Vector &b,
		Vector &results)
{
	int row = results.size();
	int col = x.size();
	
	for(int i = 0; i < row; i++)
	{
		results[i] = b[i];
	}
	
	for(int i = 0; i < row; i++)
	{
		for(int j = 0; j < col; j++)
		{
			results[i] += A[i][j] * x[j];
		}
	}
}

void xdotA(Vector &x, Matrix &A, Vector &results)
{
	int row = x.size();
	int col = results.size();
		
	for (int j = 0; j < col; ++j)
	{
		results[j] = 0;
	}

	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			results[j] += x[i] * A[i][j];
		}
	}
}

void A_add_xTmulty(Vector &x, Vector &y, Matrix &A)
{
	int row = x.size();
	int col = y.size();

	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			A[i][j] += x[i] * y[j];
		}
	}
}

double dotProduct(Vector &x, Vector &y)
{
	double dotP = 0;
	for(int i = 0; i < x.size(); i++)
	{
		dotP += x[i] * y[i];
	}
	return dotP;
}

#endif