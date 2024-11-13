#include<mpi.h>
#include<stdio.h>
#include<string.h>

void main(int argc,char *argv[])
{
	MPI_Init(&argc,&argv);
  	int dim=2;
  	int dimv[2]={2,2};
  	int period[2]={1,1};
  	int reorder=1;
  	MPI_Comm Cart_Topology;
  	MPI_Cart_create(MPI_COMM_WORLD,dim,dimv,period,reorder,&Cart_Topology);
	int rank;
	MPI_Comm_rank(Cart_Topology,&rank);
	int A[4][4],B[4][4],C[4][4], A_local[2][2], B_local[2][2], C_local[2][2];
	if(rank==0)
	{	
		int A_temp[4][4] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
		int B_temp[4][4] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
		int C_temp[4][4] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	 	memcpy(A, A_temp, sizeof(A_temp));
		memcpy(B, B_temp, sizeof(B_temp));
		memcpy(C, C_temp, sizeof(C_temp));
	}
	int sendcounts1[4] = {2,2,2,2};
	int displs1[4] = {0,2,8,10};
	MPI_Scatterv(&A, sendcounts1, displs1, MPI_INT, &A_local[0], 2, MPI_INT, 0, Cart_Topology);
	MPI_Scatterv(&B, sendcounts1, displs1, MPI_INT, &B_local[0], 2, MPI_INT, 0, Cart_Topology);
	int sendcounts2[4] = {2,2,2,2};
	int displs2[4] = {4,6,12,14};
	MPI_Scatterv(&A, sendcounts2, displs2, MPI_INT, &A_local[1], 2, MPI_INT, 0, Cart_Topology);
	MPI_Scatterv(&B, sendcounts2, displs2, MPI_INT, &B_local[1], 2, MPI_INT, 0, Cart_Topology);
	for(int step=0;step<2;step++)
	{       if(step==0)

		{if (rank==2||rank==3)
		{
		MPI_Request send_request, recv_request;
		MPI_Status status;		
		int left,right;
  		int shift=-1;
  		int change=1;
 		MPI_Cart_shift(Cart_Topology,change,shift,&left,&right);
		MPI_Isend(&A_local,4,MPI_INT,left,0,Cart_Topology,&send_request);
		MPI_Irecv(&A_local,4,MPI_INT,right,0,Cart_Topology,&recv_request);
		MPI_Wait(&send_request, &status);
		MPI_Wait(&recv_request, &status);
		}	
		if (rank==1||rank==3)
		{
		MPI_Request send_request1, recv_request1;
		MPI_Status status1;		
		int up,down;
  		int shift=1;
  		int change=0;
 		MPI_Cart_shift(Cart_Topology,change,shift,&up,&down);
		MPI_Isend(&B_local,4,MPI_INT,up,1,Cart_Topology,&send_request1);
		MPI_Irecv(&B_local,4,MPI_INT,down,1,Cart_Topology,&recv_request1);
		MPI_Wait(&send_request1, &status1);
		MPI_Wait(&recv_request1, &status1);
		}
		}
	
	if(step!=0)
	    	{
		MPI_Request send_request2, recv_request2;
		MPI_Status status2;		
		int left1,right1;
  		int shift1=-1;
  		int change1=1;
 		MPI_Cart_shift(Cart_Topology,change1,shift1,&left1,&right1);
		MPI_Isend(&A_local,4,MPI_INT,left1,0,Cart_Topology,&send_request2);
		MPI_Irecv(&A_local,4,MPI_INT,right1,0,Cart_Topology,&recv_request2);
		MPI_Wait(&send_request2, &status2);
		MPI_Wait(&recv_request2, &status2);

		MPI_Request send_request3, recv_request3;
		MPI_Status status3;		
		int up1,down1;
  		int shift2=1;
  		int change2=0;
 		MPI_Cart_shift(Cart_Topology,change2,shift2,&up1,&down1);
		MPI_Isend(&B_local,4,MPI_INT,up1,1,Cart_Topology,&send_request3);
		MPI_Irecv(&B_local,4,MPI_INT,down1,1,Cart_Topology,&recv_request3);
		MPI_Wait(&send_request3, &status3);
		MPI_Wait(&recv_request3, &status3);
		}

	for(int i=0;i<2;i++)
 	   {	for (int j=0;j<2;j++)
		{ for (int k=0;k<2;k++)
			{C_local[i][j] =C_local[i][j]+ A_local[i][k]*B_local[k][j];}
		}
	   }
	}

	MPI_Gatherv(&C_local[0],2, MPI_INT,&C,sendcounts1, displs1, MPI_INT, 0, Cart_Topology);
	MPI_Gatherv(&C_local[1],2, MPI_INT,&C,sendcounts2, displs2, MPI_INT, 0, Cart_Topology);


	if (rank==0)
	{
		for(int i=0;i<4;i++)
 		{for (int j=0;j<4;j++)
		{printf("%d ",C[i][j]);}
		printf("\n");}
        }
	
	

	
	MPI_Finalize();
}