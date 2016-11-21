// Distributed two-dimensional Discrete FFT transform
// Anisha Gartia
// ECE6122 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "Complex.h"
#include "InputImage.h"

using namespace std;

void Transform1D(Complex* h, int w, Complex* H);
void TransposeIm(Complex* H, int w, int ht, Complex* H_tr);
void InverseTransform1D (Complex *h, int w, Complex* H);


// -----------------------------------------------------------
// Function: Transform2D
// Parameters: 
// inputFN - Reference to Input image filename of type char.
// H_2dfft - Refernce to Required 2D DFT matrix of type complex.
// debug - if 1, prints all message passing events on terminal
// -----------------------------------------------------------
void Transform2D(const char* inputFN, Complex* H_2dfft, int debug) 
{
  int nCPUs, rank, rc;

  MPI_Comm_size( MPI_COMM_WORLD, &nCPUs);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);

  Complex buf[nCPUs];  // Message contents
  Complex recvbuf[nCPUs];
  Complex buf_tr[nCPUs*nCPUs];
  Complex recvbuf_tr[nCPUs*nCPUs];

  
  printf ("Number of tasks= %d My rank= %d\n", nCPUs, rank); 
  string out1dfn("MyAfter1D.txt");
  
  InputImage image(inputFN);

  int w = image.GetWidth();
  int ht = image.GetHeight();  
  int nRows = ht; 
  Complex *H = new Complex[(w * ht)];
  Complex *h = new Complex[(w * ht)];
  h = image.GetImageData();
  int rowspercpu = nRows/nCPUs;
  int colspercpu = w/nCPUs;
  

  //cout << "h real " << h[(rownum * N) + k].real << " himag " << h[(rownum * N) + k] << endl; 
  for (int rownum =0; rownum < rowspercpu; rownum++){
    int rw = rank*rowspercpu + rownum;
    // 1D Transform of each row for current CPU
    Transform1D((h + (rw*w)) , w, (H+(rw*w)));
    if ((debug == 1)) {cout << "Transform of row " << rw << " completed by CPU " << rank << endl;}
    // Send computed data to remaining CPUs. In Iteration 0, no data is sent or recieved.
    for (int cpuitr = 0; cpuitr < nCPUs; cpuitr++){
      if (cpuitr == 0){
        continue; }
      int destcpunum = rank + cpuitr;
      if (destcpunum >= nCPUs)
        {destcpunum = destcpunum - nCPUs;} 
      if ((debug == 1)) {cout << "rank " << rank << " cpuitr "<< cpuitr <<" dest cpu num " << destcpunum  << endl;}
      // Create array of all values to be sent to destcpunum
      for (int colnum = 0; colnum < colspercpu; colnum++){
        buf[colnum] = H[(rw * w) + (colnum * colspercpu) + destcpunum];
      }
      if ((debug == 1)) {cout << "rank " << rank << " buf_size " << sizeof(buf)  << endl;}
      // Blocking Send
      rc = MPI_Send(buf, sizeof(buf), MPI_CHAR, destcpunum , rownum, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) {
        cout << "Rank " << rank << " send failed, rc " << rc << endl;
        MPI_Finalize();
        exit(1);
      }
      if ((debug == 1)) {cout << "rank " << rank << " Message sent to CPU "<< destcpunum << " rc " << rc << endl;}

      //Non Blocking Recieve
      MPI_Status status;	
      MPI_Request request;
      rc = MPI_Irecv(recvbuf, sizeof(recvbuf), MPI_CHAR, MPI_ANY_SOURCE, rownum, MPI_COMM_WORLD, &request);
      if (rc != MPI_SUCCESS) {
        cout << "Rank " << rank << " recv failed, rc " << rc << endl;
        MPI_Finalize();
        exit(1);
      }
      int count = 0;
      MPI_Wait(&request, &status);
      MPI_Get_count(&status, MPI_CHAR, &count);
      cout << "Rank " << rank << " received " << count << " bytes from " << status.MPI_SOURCE << " cpuitr " << cpuitr << endl;
      int recvrw = (status.MPI_SOURCE * rowspercpu) + rownum;
      for (int colnum = 0; colnum < colspercpu; colnum++){
        H[(recvrw * w) + (colnum * colspercpu) + rank] = recvbuf[colnum];
      }      
    }
  }
  if (debug == 1) {cout << "\n" << " RANK " << rank << " ALL ROWS 1D TRANSFORM COMPLETE" << "\n" << endl;}
  string temp1d("temp1d.txt");
  
  // Transpose H to perform 1D FFT Transform of columns.
  Complex *H_tr = new Complex[(w * ht)];
  Complex *H_2d = new Complex[(w * ht)];
  TransposeIm(H, w, ht, H_tr);   
  if ((debug == 1) && (rank == 0)) {cout << "rank " << rank << "Transpose Complete " << endl;}
  for (int rowitr =0; rowitr < colspercpu; rowitr++){
    int rw = (rowitr * nCPUs) + rank;
    // 1D Transform of each row for current CPU
    Transform1D((H_tr + (rw * w)), w , (H_2d + (rw * w)));
  
    //if ((debug == 1)) {cout << "rank " << rank << " Row "<< rw << endl;}
    if (rank != 0){
      // Create array of all values to be sent to CPU 0
      for (int colnum = 0; colnum < ht; colnum++){
        buf_tr[colnum] = H_2d[(rw * ht) + colnum];
        //if ((debug == 1) && (rank == 5)) {cout << "rank " << rank << " row " << rw << " buf[" << colnum << "] = " << buf_tr[colnum] <<  endl;}
      }
      // Blocking Send
      rc = MPI_Send(buf_tr, sizeof(buf_tr), MPI_CHAR, 0 , rowitr, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) {
        cout << "Rank " << rank << " send failed, rc " << rc << endl;
        MPI_Finalize();
        exit(1);
      }
    } 
    //Non Blocking Recieve at CPU0 from all other CPUs.
    if (rank == 0){
      for (int cpu_itr = 0; cpu_itr < nCPUs; cpu_itr++){
        if (cpu_itr == 0){
          continue; }
        MPI_Status status;	
        MPI_Request request;
        rc = MPI_Irecv(recvbuf_tr, sizeof(recvbuf_tr), MPI_CHAR, MPI_ANY_SOURCE, rowitr, MPI_COMM_WORLD, &request);
        if (rc != MPI_SUCCESS) {
          cout << "Rank " << rank << " recv failed, rc " << rc << endl;
          MPI_Finalize();
          exit(1);
        }
        int count = 0;
        MPI_Wait(&request, &status);
        MPI_Get_count(&status, MPI_CHAR, &count);
        cout << "Rank " << rank << " received " << count << " bytes from " << status.MPI_SOURCE << endl;
        int recvrw = (rowitr * nCPUs) + status.MPI_SOURCE;
        for (int colnum = 0; colnum < ht; colnum++){
          H_2d[(recvrw * ht) + colnum] = recvbuf_tr[colnum];
        }      
      }
    }
  }
  if (rank == 0){
    TransposeIm(H_2d, w, ht, H_2dfft);   
  }
}


//**************

// -----------------------------------------------------------
// Function: InverseTransform2D
// Parameters: 
// inputFN - Reference to Input image filename of type char.
// H - Refernce to Input 2D DFT matrix of type complex.
// h_2difft - Refernce to Required 2D IDFT matrix of type complex.
// debug - if 1, prints all message passing events on terminal
// -----------------------------------------------------------
void InverseTransform2D(const char* inputFN, Complex* H, Complex* h_2difft, int debug) 
{
  int nCPUs, rank, rc;

  MPI_Comm_size( MPI_COMM_WORLD, &nCPUs);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);

  Complex buf[nCPUs];  // Message contents
  Complex recvbuf[nCPUs];
  Complex buf_tr[nCPUs*nCPUs];
  Complex recvbuf_tr[nCPUs*nCPUs];

  
  printf ("Number of tasks= %d My rank= %d\n", nCPUs, rank); 
  
  InputImage image(inputFN);

  int w = image.GetWidth();
  int ht = image.GetHeight();  
  int nRows = ht; 
  Complex *h = new Complex[(w * ht)];
  int rowspercpu = nRows/nCPUs;
  int colspercpu = w/nCPUs;
  

  //cout << "h real " << h[(rownum * N) + k].real << " himag " << h[(rownum * N) + k] << endl; 
  for (int rownum =0; rownum < rowspercpu; rownum++){
    int rw = rank*rowspercpu + rownum;
    // 1D Transform of each row for current CPU
    InverseTransform1D((H + (rw*w)) , w, (h+(rw*w)));
    if ((debug == 1)) {cout << "Transform of row " << rw << " completed by CPU " << rank << endl;}
    // Send computed data to remaining CPUs. In Iteration 0, no data is sent or recieved.
    for (int cpuitr = 0; cpuitr < nCPUs; cpuitr++){
      if (cpuitr == 0){
        continue; }
      int destcpunum = rank + cpuitr;
      if (destcpunum >= nCPUs)
        {destcpunum = destcpunum - nCPUs;} 
      if ((debug == 1)) {cout << "rank " << rank << " cpuitr "<< cpuitr <<" dest cpu num " << destcpunum  << endl;}
      // Create array of all values to be sent to destcpunum
      for (int colnum = 0; colnum < colspercpu; colnum++){
        buf[colnum] = h[(rw * w) + (colnum * colspercpu) + destcpunum];
      }
      if ((debug == 1)) {cout << "rank " << rank << " buf_size " << sizeof(buf)  << endl;}
      // Blocking Send
      rc = MPI_Send(buf, sizeof(buf), MPI_CHAR, destcpunum , rownum, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) {
        cout << "Rank " << rank << " send failed, rc " << rc << endl;
        MPI_Finalize();
        exit(1);
      }
      if ((debug == 1)) {cout << "rank " << rank << " Message sent to CPU "<< destcpunum << " rc " << rc << endl;}

      //Non Blocking Recieve
      MPI_Status status;	
      MPI_Request request;
      rc = MPI_Irecv(recvbuf, sizeof(recvbuf), MPI_CHAR, MPI_ANY_SOURCE, rownum, MPI_COMM_WORLD, &request);
      if (rc != MPI_SUCCESS) {
        cout << "Rank " << rank << " recv failed, rc " << rc << endl;
        MPI_Finalize();
        exit(1);
      }
      int count = 0;
      MPI_Wait(&request, &status);
      MPI_Get_count(&status, MPI_CHAR, &count);
      cout << "Rank " << rank << " received " << count << " bytes from " << status.MPI_SOURCE << " cpuitr " << cpuitr << endl;
      int recvrw = (status.MPI_SOURCE * rowspercpu) + rownum;
      for (int colnum = 0; colnum < colspercpu; colnum++){
        h[(recvrw * w) + (colnum * colspercpu) + rank] = recvbuf[colnum];
      }      
    }
  }
  if (debug == 1) {cout << "\n" << " RANK " << rank << " ALL ROWS 1D TRANSFORM COMPLETE" << "\n" << endl;}
  string temp1d("temp1d.txt");
  
  // Transpose H to perform 1D FFT Transform of columns.
  Complex *h_tr = new Complex[(w * ht)];
  Complex *h_2d = new Complex[(w * ht)];
  TransposeIm(h, w, ht, h_tr);   
  if ((debug == 1) && (rank == 0)) {cout << "rank " << rank << "Transpose Complete " << endl;}
  for (int rowitr =0; rowitr < colspercpu; rowitr++){
    int rw = (rowitr * nCPUs) + rank;
    // 1D Transform of each row for current CPU
    InverseTransform1D((h_tr + (rw * w)), w , (h_2d + (rw * w)));
  
    //if ((debug == 1)) {cout << "rank " << rank << " Row "<< rw << endl;}
    if (rank != 0){
      // Create array of all values to be sent to CPU 0
      for (int colnum = 0; colnum < ht; colnum++){
        buf_tr[colnum] = h_2d[(rw * ht) + colnum];
        //if ((debug == 1) && (rank == 5)) {cout << "rank " << rank << " row " << rw << " buf[" << colnum << "] = " << buf_tr[colnum] <<  endl;}
      }
      // Blocking Send
      rc = MPI_Send(buf_tr, sizeof(buf_tr), MPI_CHAR, 0 , rowitr, MPI_COMM_WORLD);
      if (rc != MPI_SUCCESS) {
        cout << "Rank " << rank << " send failed, rc " << rc << endl;
        MPI_Finalize();
        exit(1);
      }
    } 
    //Non Blocking Recieve at CPU0 from all other CPUs.
    if (rank == 0){
      for (int cpu_itr = 0; cpu_itr < nCPUs; cpu_itr++){
        if (cpu_itr == 0){
          continue; }
        MPI_Status status;	
        MPI_Request request;
        rc = MPI_Irecv(recvbuf_tr, sizeof(recvbuf_tr), MPI_CHAR, MPI_ANY_SOURCE, rowitr, MPI_COMM_WORLD, &request);
        if (rc != MPI_SUCCESS) {
          cout << "Rank " << rank << " recv failed, rc " << rc << endl;
          MPI_Finalize();
          exit(1);
        }
        int count = 0;
        MPI_Wait(&request, &status);
        MPI_Get_count(&status, MPI_CHAR, &count);
        cout << "Rank " << rank << " received " << count << " bytes from " << status.MPI_SOURCE << endl;
        int recvrw = (rowitr * nCPUs) + status.MPI_SOURCE;
        for (int colnum = 0; colnum < ht; colnum++){
          h_2d[(recvrw * ht) + colnum] = recvbuf_tr[colnum];
        }      
      }
    }
  }
  if (rank == 0){
    TransposeIm(h_2d, w, ht, h_2difft);   
  }
}



// -----------------------------------------------------------
// Function: Transform1D
// Performs 1D tranformation of a given row
// Parameters: 
// h - Reference to Input image matrix of type Complex
// w - width of input image referenced by H
// H - Reference to image matrix with the required tranformed row.
// -----------------------------------------------------------

void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  int N = w;
  //for (int row =0; row < nRows; row++){
    //cout << "row " << rownum << endl;
    for(int n=0; n < N; n++){ 
      Complex sum_Hn(0,0);
      for(int k=0; k < N; k++){
        Complex W (cos(2*M_PI*n*k/N) , -sin(2*M_PI*n*k/N));
        sum_Hn = sum_Hn + W * h[k] ;
      }
      H[n] = sum_Hn;
    }
}



void InverseTransform1D(Complex* h, int w, Complex* H)
{
  int N = w;
  for(int n=0; n < N; n++){ 
    Complex sum_Hn(0,0);
    for(int k=0; k < N; k++){
      Complex W (cos(2*M_PI*n*k/N) , sin(2*M_PI*n*k/N));
      sum_Hn = sum_Hn + W * h[k] ;
    }
    H[n] = Complex((1.0/(N)),0) * sum_Hn;
  }
}


// -----------------------------------------------------------
// Function: TransposeIm
// Parameters: 
// H - Reference to Input image matrix of type Complex
// w - width of input image referenced by H
// ht - height of input image referenced by H
// H_tr - Required transpose
// -----------------------------------------------------------
void TransposeIm(Complex* H, int w, int ht, Complex* H_tr)
{
  for (int row_itr = 0; row_itr < ht; row_itr++){
    for (int col_itr = 0; col_itr < w; col_itr++){
      H_tr[(col_itr * ht) + row_itr]  = H[(row_itr * w) + col_itr];
    }
  }
}      

int main(int argc, char** argv)
{
  int  nCPUs, rank, rc; 
  int debug = 0; // if 1, prints all message passing events onto console.

  rc = MPI_Init( &argc, &argv);
  if (rc != MPI_SUCCESS) {    printf ("Error starting MPI program. Terminating.\n");
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  MPI_Comm_size( MPI_COMM_WORLD, &nCPUs);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank);
  
  string fn("Tower.txt"); // default file name
  string outfn("MyAfter2D.txt");
  string invoutfn("MyAfterInverse.txt");
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  InputImage imTower(fn.c_str());

  int w = imTower.GetWidth();
  int ht = imTower.GetHeight();  
  Complex *H_2dfft = new Complex[(w * ht)];
  Transform2D(fn.c_str(), H_2dfft, debug);
  if (rank == 0){
    cout << "2D transform computed. Matrix size " << sizeof(*H_2dfft) << ".  Saving to file" << endl;
    imTower.SaveImageData(outfn.c_str(), H_2dfft, w, ht);
  }

  // Inverse FFT
  // Send all computed 2D FFT values to from CPU 0 to all other CPUs.
  MPI_Request request;
  MPI_Status status;
  Complex buf_H_2dfft[(w*ht)];	  
  if (rank == 0){
    for (int i = 0; i < (w*ht); i++)
	{ buf_H_2dfft[i] = H_2dfft[i];}
    for (int cpunum =0; cpunum < nCPUs; cpunum++){
      // Non Blocking send from rank 0 to other CPUs. 
      if (cpunum == 0)
        {continue;}
      cout << "Rank " << rank << " sending complete 2D DFT Matrix to rank" << cpunum << " Sending bytes " << sizeof(buf_H_2dfft) << endl;
      MPI_Status status;	  
      rc = MPI_Isend(buf_H_2dfft, sizeof(buf_H_2dfft), MPI_CHAR, cpunum, cpunum, MPI_COMM_WORLD, &request);
      if (rc != MPI_SUCCESS) {
        cout << "Rank " << rank << " send failed, rc " << rc << endl;
        MPI_Finalize();
        exit(1);
      } 
      MPI_Wait(&request, &status);
    }
  }
  else {
    rc = MPI_Irecv(buf_H_2dfft, sizeof(buf_H_2dfft), MPI_CHAR, MPI_ANY_SOURCE, rank, MPI_COMM_WORLD, &request);
    if (rc != MPI_SUCCESS) {
      cout << "Rank " << rank << " recv failed, rc " << rc << endl;
      MPI_Finalize();
      exit(1);
    } 
    int count = 0;
    MPI_Wait(&request, &status);
    MPI_Get_count(&status, MPI_CHAR, &count);
    cout << "Rank " << rank << "receiving complete 2D DFT Matrix from rank " << status.MPI_SOURCE << endl;
    cout << "Rank " << rank  << " received " << count << " bytes from " << status.MPI_SOURCE << endl;
  }
  Complex *h_2difft = new Complex[(w * ht)];
  for (int i = 0; i < (w*ht); i++)
    {  H_2dfft[i] = buf_H_2dfft[i]; }
  InverseTransform2D(fn.c_str(), H_2dfft, h_2difft,  debug);
  if (rank == 0){
    cout << "2D inverse transform computed. Saving to file" << endl;
    imTower.SaveImageDataReal(invoutfn.c_str(), h_2difft, w, ht);}
  MPI_Finalize();

}  
  


