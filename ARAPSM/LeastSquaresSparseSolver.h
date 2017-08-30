// ---------------------------------------------------------------------------------------------------------
// Please see below for using the linear solver. It provides Conjugate and Cholosky approaches.
// To build the linear system, you have to:
//
// 1. Create or reset the system matrix.	#solver.Create(rows , cols , dims) or #solver.ResetSolver(rows, cols , dims)
// 2. Add the non-zero element				#solver.AddSysElement(rowIdx , colIdx , value)
// 3. Set the right hand side vectors		#solver.SetRightHandSideMatrix(b)
//
// To solve the linear system using Conjugate gradient method, you then
//
// 1. Set the initial guess					#solver.SetInitialGuess(x)
// 2. Solve									#solver.ConjugateGradientSolve()
// 3. Get the result						#solver.GetSolution()
//
// To solve the linear system using Cholosky method, you then
//
// 1. Factorize the system matrix			#solver.CholoskyFactorization()
// 2. Solve									#solver.CholoskySolve()
// 3. Get the result						#solver.GetSolution()
//
// Finally, please call #solver.ResetSolver(0 , 0 , 0) to reset your memory! Enjoy yourself!
// ---------------------------------------------------------------------------------------------------------

#ifndef SPARSE_LEASTSQUARES_SOLVER_H
#define SPARSE_LEASTSQUARES_SOLVER_H

#include <omp.h>
#include <Windows.h>
#include <stdlib.h>
#include <vector>
#include <taucs.h>

#include "TimeRecorder.h"

extern "C"
{
	#include <cs.h>
};

using namespace std;
#pragma comment(lib, "CSparse.lib")

#pragma comment(lib, "libtaucs.lib")
#pragma comment(lib, "libatlas.lib")
#pragma comment(lib, "libcblas.lib")
#pragma comment(lib, "libf77blas.lib")
#pragma comment(lib, "liblapack.lib")
#pragma comment(lib, "libmetis.lib")
#pragma comment(lib, "vcf2c.lib")

class LeastSquaresSparseSolver
{
public:

	LeastSquaresSparseSolver()
	{
		A = triplet = ATA = NULL;
		x = b = ATb = NULL;
		taucsATA = NULL;
		Factorization = NULL;
		cols = rows = 0;
		Is_System_Stable = Is_RHS_Stable = false;
	}

	~LeastSquaresSparseSolver()
	{
		deallocate();
	}

	bool IsTheSameSize(int rowNum , int colNum)
	{
		if (rows == rowNum && cols == colNum)
			return true;
		else
			return false;
	}

	void Create(int rowNum , int colNum , int dimNum)
	{
		rows = rowNum;
		cols = colNum;
		dims = dimNum;
		Is_System_Stable = Is_RHS_Stable = false;

		AllocateSystemMatrix();
		AllocateRightHandSideMatrix();
	}

	void ResetSolver(int rowNum , int colNum , int dimNum)
	{
		deallocate();

		if (rowNum == 0 && colNum == 0 && dimNum == 0)
			return;

		Create(rowNum , colNum , dimNum);
	}

	void AddSysElement(int rowIdx , int colIdx , float val)
	{
		if (rowIdx >= rows || colIdx >= cols || rowIdx < 0 || colIdx < 0)
			printf("Warning! An index of the system matrix is wrong! %d %d %d %d\n", rowIdx , rows , colIdx , cols );

		cs_entry(triplet , rowIdx , colIdx , val);
		Is_System_Stable = false;
	}

	void SetRightHandSideMatrix(float **m)
	{
		b = m;
		Is_RHS_Stable = false;
	}

	float GetSolution(int d , int rowIdx)
	{
		return x[d][rowIdx];
	}

	void SetInitialGuess(float **m)
	{
		for (int d = 0 ; d < dims ; d++)
			for (int i = 0 ; i < cols ; i++)
				x[d][i] = m[d][i];
	}

	void SetInitialGuess(int d , int r , float val)
	{
		x[d][r] = val;
	}

	void matrixStablization()
	{
		if (!Is_System_Stable)		LeastSquareToGeneralSparseMatrix();
		if (!Is_RHS_Stable)			ATbComputation();
	}

	void ConjugateGradientSolve(float epsilon = 0.001f , int max_iter = 5000 , bool silence = true)
	{
		TimeRecorder timer;

		if (!silence)
		{
			timer.ResetTimer();
			printf("Start solving the system, with %d rows and %d cols... ", rows , cols);
		}

		matrixStablization();								//OutputLinearSystem("ATA.mtx" , "rhs.vec");
		ConjugateGradient(epsilon , max_iter);				//OutputSolution("Solution.vec");

		if (!silence)
			printf("passed time: %f\n" , timer.PassedTime());
	}

	bool CholoskyFactorization(bool silence = true)
	{
		TimeRecorder timer;

		if (!silence)
		{
			timer.ResetTimer();
			printf("Start solving the system, with %d rows and %d cols... ", rows , cols);
		}
		
		matrixStablization();

		// Convert to lower triangular Taucs format
		int m = ATA->m;
		int nnz = (ATA->p[m] - m) / 2 + m;
		taucsATA = taucs_ccs_create(m , m , nnz , TAUCS_SINGLE | TAUCS_LOWER | TAUCS_SYMMETRIC);

		int pIdx = 0;
		int *AP = ATA->p , *AI = ATA->i;
		float *AX = ATA->x;

		for (int i = 0 ; i < m ; i++)
		{
			taucsATA->colptr[i] = pIdx;

			for (int j = AP[i] ; j < AP[i + 1] ; j++)
			{
				if (i > AI[j])		continue;

				taucsATA->rowind[pIdx] = AI[j];
				taucsATA->values.s[pIdx++] = AX[j];
			}
		}

		taucsATA->colptr[m] = pIdx;

		// -------------------- Important!!! ------------------------
		// Once you choose to solve the linear system using Taucs,
		// you cannot go back to use CG solver again unless you
		// rebuild the linear system.
		// ----------------------------------------------------------
		cs_spfree(ATA);

		// factorization
		// ---------------------------------- In-Core --------------------------------------------------
		char* factor_opt[] = { "taucs.factor.LLT=true", NULL };
		int message = taucs_linsolve(taucsATA , &Factorization , 0 , NULL , NULL , factor_opt , NULL);
		ShowTaucsErrorMessage(message , 0);
		// ---------------------------------------------------------------------------------------------

		//// ------------------------------ Out-of-Core ------------------------------------------------
		//double memroy = taucs_available_memory_size();

		//char* factor_opt[] = {  "taucs.factor.LLT=true" ,
		//						"taucs.ooc=true" ,
		//						"taucs.ooc.basename=taucs" , 
		//						"taucs.ooc.memory=#0", NULL};

		//void* argu[] = {&memroy , NULL};

		//taucs_linsolve( taucsATA, &Factorization, 0, NULL, NULL, factor_opt, argu );
		//// -------------------------------------------------------------------------------------------

		if (!silence)
			printf("passed time: %f\n" , timer.PassedTime());

		if (message != 0)		return false;
		else					return true;
	}

	void CholoskySolve(int d)
	{
		if (d >= dims)
		{
			printf("dimension error!\n");
			return;
		}
		
		matrixStablization();

		// ---------------------------------- In-Core --------------------------------------------------
		char* solve_opt [] = { "taucs.factor=false", NULL };
		int message = taucs_linsolve(taucsATA , &Factorization , 1 , x[d] , ATb[d] , solve_opt , NULL);
		ShowTaucsErrorMessage(message , 1);
		// ---------------------------------------------------------------------------------------------

		//// ------------------------------ Out-of-Core ------------------------------------------------
		//double memroy = taucs_available_memory_size();

		//char* factor_opt[] = {  "taucs.factor.LLT=false" ,
		//						"taucs.ooc=true" ,
		//						"taucs.ooc.basename=taucs" ,
		//						"taucs.ooc.memory=#0" , NULL};

		//void* argu[] = {&memroy , NULL};

		//taucs_linsolve( taucsATA , &Factorization , 1 , x , b , factor_opt , argu );
		//// -------------------------------------------------------------------------------------------
	}

private:
	int dims , cols , rows;
	bool Is_System_Stable , Is_RHS_Stable;

	// TAUCS matrix
	void *Factorization;
	taucs_ccs_matrix *taucsATA;

	// Sparse matrix
	cs *triplet , *A , *ATA;

	// Dense vector
	float **x , **b , **ATb;

	void deallocate()
	{
		if (x != NULL)
		{
			for (int i = 0 ; i < dims ; i++)
				delete [] x[i];

			delete [] x;
			x = NULL;
		}

		if (ATb != NULL)
		{
			for (int i = 0 ; i < dims ; i++)
				delete [] ATb[i];

			delete [] ATb;
			ATb = NULL;
		}

		if (A != NULL)
		{
			cs_spfree(A);
			A = NULL;
		}

		if (triplet != NULL)
		{
			cs_spfree(triplet);
			triplet = NULL;
		}

		// Taucs
		if (taucsATA != NULL)
		{
			taucs_ccs_free(taucsATA);
			taucsATA = NULL;
		}

		if (Factorization != NULL)
		{
			taucs_linsolve(NULL , &Factorization , 0 , NULL , NULL , NULL , NULL);
			Factorization = NULL;
		}
	}

	void ShowTaucsErrorMessage(int message , int state)
	{
		static const char* str_state[2] =
		{
			"Factorization step:", "Substitution:step"
		};

		static const char* str_error[7] = 
		{
			"TAUCS_SUCCESS", "TAUCS_ERROR", "TAUCS_ERROR_NOMEM", "TAUCS_ERROR_BADARGS", 
			"TAUCS_ERROR_INDEFINITE", "TAUCS_ERROR_MAXDEPTH", "NON_DEFINE"
		};

		if (message != 0)
		{
			//cs_print(A , 0);
			printf("%s %s\n" , str_state[state] , str_error[-message]);
		}
	}

	void OutputLinearSystem(char *sys , char *rhs)
	{
		// output system matrix
		FILE *output = fopen(sys , "w");

		int *AP = ATA->p , *AI = ATA->i;
		float *AX = ATA->x;

		for (int i = 0 ; i < ATA->m ; i++)
		{
			for (int j = AP[i] ; j < AP[i + 1] ; j++)
				fprintf(output , "%d %d %f\n" , i + 1 , AI[j] + 1 , AX[j]);
		}


		fclose(output);

		// output right hand side vector
		output = fopen(rhs , "w");

		for (int d = 0 ; d < dims ; d++)
			for (int i = 0 ; i < cols ; i++)
				fprintf(output , "%f\n" , ATb[d][i]);

		fclose(output);
	}

	void OutputSolution(char *solution)
	{
		// output X

		FILE *output = fopen(solution , "w");

		for (int d = 0 ; d < dims ; d++)
			for (int i = 0 ; i < cols ; i++)
				fprintf(output , "%f\n" , x[d][i]);

		fclose(output);
	}

	void AllocateSystemMatrix()
	{
		x = new float*[dims];

		for (int d = 0 ; d < dims ; d++)
			x[d] = new float[cols];

		for (int d = 0 ; d < dims ; d++)
			for (int i = 0 ; i < cols ; i++)
				x[d][i] = 0.0f;
	
		triplet = cs_spalloc(rows , cols , rows , 1 , 1);
	}

	void AllocateRightHandSideMatrix()
	{
		ATb = new float*[dims];

		for (int d = 0 ; d < dims ; d++)
			ATb[d] = new float[cols];
	}

	void LeastSquareToGeneralSparseMatrix()
	{
		cs *AT;
		A = cs_compress(triplet);

		// ------------------------- Important !!! ----------------------------
		// You cannot add triplet element any more once you convert the
		// matrix to a general format, unless you rebuild the linear system!
		// --------------------------------------------------------------------
		cs_spfree(triplet);
		triplet = NULL;

		AT = cs_transpose(A , 1);
		ATA = cs_multiply(AT , A);

		// release AT
		cs_spfree(AT);

		Is_System_Stable = true;
	}

	//void LeastSquareToGeneralSparseMatrix()
	//{
	//	cs *AT;
	//	A = cs_compress(triplet);

	//	// ------------------------- Important !!! ----------------------------
	//	// You cannot add triplet element any more once you convert the
	//	// matrix to a general format, unless you rebuild the linear system!
	//	// --------------------------------------------------------------------
	//	cs_spfree(triplet);
	//	triplet = NULL;

	//	AT = cs_transpose(A , 1);
	//	
	//	vector<vector<int> > rowIdxList;
	//	rowIdxList.resize(AT->m);

	//	for (int i = 0 ; i < AT->m ; i++)
	//		for (int j = AT->p[i] ; j <= AT->p[i + 1] ; j++)
	//			rowIdxList[AT->i[i]].push_back(i);


	//	// release AT
	//	cs_spfree(AT);

	//	Is_System_Stable = true;
	//}

	void ATbComputation()
	{
		if (b == NULL)
			printf("The right hand side vector is NULL!\n");

		int *AP = A->p , *AI = A->i;
		float *AX = A->x;
		
		for (int d = 0 ; d < dims ; d++)
		{
			#pragma omp parallel for
			for (int i = 0 ; i < A->n ; i++)
			{
				ATb[d][i] = 0.0f;

				for (int j = AP[i] ; j < AP[i + 1] ; j++)
					ATb[d][i] += AX[j] * b[d][AI[j]];			// row = AI[j];	 col = i;  val = AX[j];
			}
		}

		Is_RHS_Stable = true;
	}

	void ConjugateGradient(float epsilon = 0.001f , int max_iter = 5000)
	{
		float *r = new float[cols];
		float *d = new float[cols];
		float *Ad = new float[cols];
		float *h = new float[cols];
		float *diag_inv = new float[cols];

		for (int i = 0 ; i < cols ; i++)
			r[i] = d[i] = Ad[i] = h[i] = diag_inv[i] = 0.0f;

		// diagonal inverse
		for (int i = 0 ; i < ATA->n ; i++)
			for (int j = ATA->p[i] ; j < ATA->p[i + 1] ; j++)
			{
				if (ATA->i[j] == i)
				{
					diag_inv[i] = ATA->x[j];
					break;
				}
				else if (ATA->i[j] > i)
					break;
			}

		for(int i = 0; i < cols ; i++)
			diag_inv[i] = (float)((i >= cols || diag_inv[i] == 0.0) ? 1.0 : 1.0 / diag_inv[i]) ;

		// r = A*x			//cs_gaxpy(ATA , x , r);
		MatVecMult(ATA , x[0] , r);

		// r = b - A*x
		VecAdd(r , -1.0f , ATb[0] , 1.0f , cols);

		// d = M-1 * r
		VecVecMult(diag_inv , r , d , cols);

		// cur_err = rT*d
		float cur_err = VecVecDot(r , d , cols);

		// err
		float err = (float)(cols * epsilon * epsilon);

		int its = 0;
		while ( cur_err > err && (int)its < max_iter) 
		{
			// Ad = A*d		//cs_gaxpy(ATA , d , Ad);
			memset(Ad , 0 , cols * sizeof(float));
			MatVecMult(ATA , d , Ad);

			// alpha = cur_err / (dT*Ad)
			float alpha = cur_err / VecVecDot(d , Ad , cols);

			// x = x + alpha * d
			VecAdd(x[0] , 1.0f , d , alpha , cols);

			// r = r - alpha * Ad
			VecAdd(r , 1.0f , Ad , -alpha , cols);

			// h = M-1r
			VecVecMult(diag_inv , r , h , cols);

			float old_err = cur_err ;

			// cur_err = rT * h
			cur_err = VecVecDot(r , h , cols);

			float beta = cur_err / old_err ;

			// d = h + beta * d
			VecAdd(d , beta , h , 1.0f , cols);
			
			its++;
		}

		delete [] r;
		delete [] d;
		delete [] Ad;
		delete [] h;
		delete [] diag_inv;
	}

	void MatVecMult(cs *A , float *x , float *b)
	{
		int *AP = A->p , *AI = A->i;
		float *AX = A->x;

		#pragma omp parallel for
		for (int i = 0 ; i < A->m ; i++)
		{
			b[i] = 0.0f;

			for (int j = AP[i] ; j < AP[i + 1] ; j++)
				b[i] += AX[j] * x[AI[j]];			// row = AI[j];	 col = i;  val = AX[j];
		}
	}

	void VecAdd(float *a , float alpha , float *b , float beta , int size)	// a = alpha * a + beta * b
	{
		#pragma omp parallel for
		for (int i = 0 ; i < size ; i++)
			a[i] = a[i] * alpha + b[i] * beta;
	}

	void VecVecMult(float *a , float *b , float *c , int size)				// c = a * b
	{
		#pragma omp parallel for
		for (int i = 0 ; i < size ; i++)
			c[i] = a[i] * b[i];
	}

	float VecVecDot(float *a , float *b , int size)							// return a \cdot b
	{
		float sum = 0.0f;

		for (int i = 0 ; i < size ; i++)
			sum += a[i] * b[i];

		return sum;
	}
};

#endif