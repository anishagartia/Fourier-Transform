# Fourier-Transform

Given a one–dimensional array of complex or real input values of length N, the Discrete Fourier Transform consists af an array of size N computed using the formula of Fourier Transform. Given a two–dimensional matrix of complex input values, the two–dimensional Fourier Transform can be com- puted with two simple steps. First, the one–dimensional transform is computed for each row in the matrix individually. Then a second one–dimensional transform is done on each column of the matrix individually. Note that the transformed values from the first step are the inputs to the second step.

If we have several CPU’s to use to compute the 2D DFT, it is easy to see how some of these steps can be done simulataneously. For example, if we are computing a 2D DFT of a 16 by 16 matrix, and if we had 16 CPUs available, we could assign each of the 16 CPU’s to compute the DFT of a given row. In this simple example, CPU 0 would compute the one–dimensional DFT for row 0, CPU 1 would compute for row 1, and so on. If we did this, the first step (computing DFT’s of the rows) should run 16 times faster than when we only used one CPU.

However, when computing the second step, we run into difficulties. When CPU 0 completes the one–dimensional DFT for row 0, it would presumably be ready compute the 1D DFT for column 0. Unfortunately, the computed results for all other columns are not available to CPU 0 easily. We can solve this problem by using message passing. After each CPU completes the first step (computing 1D DFT’s for each row), it must send the required values to the other processes using MPI. In this example, CPU 0 would send to CPU 1 the computed transform value for row 0, column 1, and send to CPU 2 the computed transform value for row 0, column 2, and so on. When each CPU has received k messages with column values (k is the total number of columns in the input set), it is then ready to compute the column DFT.

Finally, each CPU must report the final result (again using message passing) to a designated CPU responsible for collecting and printing the final transformed value. Normally, CPU 0 would be chosen for this task, but in fact any CPU could be assigned to do this.

## Implementation of code

fft2d.cc implements the transform as follows:

1. A simple one–dimensional DFT is implemented using the double summation approach in the equations of FFT.
2. MPI send and receive are used to send partial information between the 16 processes.
3. CPU at rank zero is used to collect the final transformed values from all other CPU’s, and write these results to a file called MyAfter2D.txt using the SaveImageData method in class InputImage.

After the 2D transform has been completed, we use MPI again to calculate the Inverse transform, and write the results to file MyAfterInverse.txt. We use the SaveImageDataReal function to write only the real part of the Image to results. This function writes the real part only (as the imaginary parts should be zero or near zero after the inverse.
