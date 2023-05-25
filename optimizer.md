Code that would otherwise compute:

<!-- Ko = R\(S' + B'P) -->
$$
    Ko = R^{-1}(S'+B'P)
$$
Or expanded:
$$
\left[
\begin{array}{c}
0.01 \\
\end{array}
\right]^{-1}

\left(
\left[
\begin{array}{c}
\lambda_2 \\
 - \lambda_1 \\
\lambda_4 \\
 - \lambda_3 \\
\end{array}
\right]^{T}
+
\left[
\begin{array}{c}
 - x_2 \\
x_1 \\
 - x_4 \\
x_3 \\
\end{array}
\right]^{T}

\left[
\begin{array}{cccc}
P_{1}ˏ_1 & P_{1}ˏ_2 & P_{1}ˏ_3 & P_{1}ˏ_4 \\
P_{2}ˏ_1 & P_{2}ˏ_2 & P_{2}ˏ_3 & P_{2}ˏ_4 \\
P_{3}ˏ_1 & P_{3}ˏ_2 & P_{3}ˏ_3 & P_{3}ˏ_4 \\
P_{4}ˏ_1 & P_{4}ˏ_2 & P_{4}ˏ_3 & P_{4}ˏ_4 \\
\end{array}
\right]
\right)

$$
% use these functions:

% and take this long to run:

Instead, we generate a scalarized kernel, which is parallelizeable (for larger models), and reduces the need for several intermediate vector/matrix computations.
$$
Ko = 

\left[
\begin{array}{c}
100.0 \lambda_2 - 100.0 x_2 P_{1}ˏ_1 + 100.0 x_1 P_{2}ˏ_1 - 100.0 x_4 P_{3}ˏ_1 + 100.0 x_3 P_{4}ˏ_1 \\
 - 100.0 \lambda_1 - 100.0 x_2 P_{1}ˏ_2 + 100.0 x_1 P_{2}ˏ_2 + 100.0 x_3 P_{4}ˏ_2 - 100.0 x_4 P_{3}ˏ_2 \\
100.0 \lambda_4 - 100.0 x_2 P_{1}ˏ_3 + 100.0 x_1 P_{2}ˏ_3 - 100.0 x_4 P_{3}ˏ_3 + 100.0 x_3 P_{4}ˏ_3 \\
 - 100.0 \lambda_3 - 100.0 x_2 P_{1}ˏ_4 + 100.0 x_1 P_{2}ˏ_4 + 100.0 x_3 P_{4}ˏ_4 - 100.0 x_4 P_{3}ˏ_4 \\
\end{array}
\right]
$$

Or alternatively

$$
\left[
\begin{array}{c}
\left( x_1 P_{2}ˏ_1 + x_3 P_{4}ˏ_1 - x_2 P_{1}ˏ_1 - x_4 P_{3}ˏ_1 + \lambda_2 \right) \left( 100.0 \lambda_2 - 100.0 x_2 P_{1}ˏ_1 + 100.0 x_1 P_{2}ˏ_1 - 100.0 x_4 P_{3}ˏ_1 + 100.0 x_3 P_{4}ˏ_1 \right) - u_1 P_{1}ˏ_2 - u_1 P_{2}ˏ_1 + P_{1}ˏ_3 + P_{3}ˏ_1 \\
\left(  - \lambda_1 + x_1 P_{2}ˏ_2 + x_3 P_{4}ˏ_2 - x_2 P_{1}ˏ_2 - x_4 P_{3}ˏ_2 \right) \left( 100.0 \lambda_2 - 100.0 x_2 P_{1}ˏ_1 + 100.0 x_1 P_{2}ˏ_1 - 100.0 x_4 P_{3}ˏ_1 + 100.0 x_3 P_{4}ˏ_1 \right) - P_{4}ˏ_1 + u_1 P_{1}ˏ_1 - u_1 P_{2}ˏ_2 + P_{2}ˏ_3 \\
\left( 100.0 \lambda_2 - 100.0 x_2 P_{1}ˏ_1 + 100.0 x_1 P_{2}ˏ_1 - 100.0 x_4 P_{3}ˏ_1 + 100.0 x_3 P_{4}ˏ_1 \right) \left( x_1 P_{2}ˏ_3 + x_3 P_{4}ˏ_3 - x_2 P_{1}ˏ_3 - x_4 P_{3}ˏ_3 + \lambda_4 \right) - P_{1}ˏ_1 - u_1 P_{3}ˏ_2 - u_1 P_{4}ˏ_1 + P_{3}ˏ_3 \\
\left(  - \lambda_3 + x_1 P_{2}ˏ_4 + x_3 P_{4}ˏ_4 - x_2 P_{1}ˏ_4 - x_4 P_{3}ˏ_4 \right) \left( 100.0 \lambda_2 - 100.0 x_2 P_{1}ˏ_1 + 100.0 x_1 P_{2}ˏ_1 - 100.0 x_4 P_{3}ˏ_1 + 100.0 x_3 P_{4}ˏ_1 \right) + u_1 P_{3}ˏ_1 - u_1 P_{4}ˏ_2 + P_{2}ˏ_1 + P_{4}ˏ_3 \\
\left( x_1 P_{2}ˏ_1 + x_3 P_{4}ˏ_1 - x_2 P_{1}ˏ_1 - x_4 P_{3}ˏ_1 + \lambda_2 \right) \left(  - 100.0 \lambda_1 - 100.0 x_2 P_{1}ˏ_2 + 100.0 x_1 P_{2}ˏ_2 + 100.0 x_3 P_{4}ˏ_2 - 100.0 x_4 P_{3}ˏ_2 \right) - P_{1}ˏ_4 + u_1 P_{1}ˏ_1 - u_1 P_{2}ˏ_2 + P_{3}ˏ_2 \\
 - P_{2}ˏ_4 + \left(  - \lambda_1 + x_1 P_{2}ˏ_2 + x_3 P_{4}ˏ_2 - x_2 P_{1}ˏ_2 - x_4 P_{3}ˏ_2 \right) \left(  - 100.0 \lambda_1 - 100.0 x_2 P_{1}ˏ_2 + 100.0 x_1 P_{2}ˏ_2 + 100.0 x_3 P_{4}ˏ_2 - 100.0 x_4 P_{3}ˏ_2 \right) + u_1 P_{1}ˏ_2 + u_1 P_{2}ˏ_1 - P_{4}ˏ_2 \\
 - P_{1}ˏ_2 + \left(  - 100.0 \lambda_1 - 100.0 x_2 P_{1}ˏ_2 + 100.0 x_1 P_{2}ˏ_2 + 100.0 x_3 P_{4}ˏ_2 - 100.0 x_4 P_{3}ˏ_2 \right) \left( x_1 P_{2}ˏ_3 + x_3 P_{4}ˏ_3 - x_2 P_{1}ˏ_3 - x_4 P_{3}ˏ_3 + \lambda_4 \right) - P_{3}ˏ_4 + u_1 P_{3}ˏ_1 - u_1 P_{4}ˏ_2 \\
 - P_{4}ˏ_4 + \left(  - 100.0 \lambda_1 - 100.0 x_2 P_{1}ˏ_2 + 100.0 x_1 P_{2}ˏ_2 + 100.0 x_3 P_{4}ˏ_2 - 100.0 x_4 P_{3}ˏ_2 \right) \left(  - \lambda_3 + x_1 P_{2}ˏ_4 + x_3 P_{4}ˏ_4 - x_2 P_{1}ˏ_4 - x_4 P_{3}ˏ_4 \right) + u_1 P_{3}ˏ_2 + u_1 P_{4}ˏ_1 + P_{2}ˏ_2 \\
 - P_{1}ˏ_1 + \left( x_1 P_{2}ˏ_1 + x_3 P_{4}ˏ_1 - x_2 P_{1}ˏ_1 - x_4 P_{3}ˏ_1 + \lambda_2 \right) \left( 100.0 \lambda_4 - 100.0 x_2 P_{1}ˏ_3 + 100.0 x_1 P_{2}ˏ_3 - 100.0 x_4 P_{3}ˏ_3 + 100.0 x_3 P_{4}ˏ_3 \right) - u_1 P_{1}ˏ_4 - u_1 P_{2}ˏ_3 + P_{3}ˏ_3 \\
 - P_{2}ˏ_1 - P_{4}ˏ_3 + \left(  - \lambda_1 + x_1 P_{2}ˏ_2 + x_3 P_{4}ˏ_2 - x_2 P_{1}ˏ_2 - x_4 P_{3}ˏ_2 \right) \left( 100.0 \lambda_4 - 100.0 x_2 P_{1}ˏ_3 + 100.0 x_1 P_{2}ˏ_3 - 100.0 x_4 P_{3}ˏ_3 + 100.0 x_3 P_{4}ˏ_3 \right) + u_1 P_{1}ˏ_3 - u_1 P_{2}ˏ_4 \\
\left( x_1 P_{2}ˏ_3 + x_3 P_{4}ˏ_3 - x_2 P_{1}ˏ_3 - x_4 P_{3}ˏ_3 + \lambda_4 \right) \left( 100.0 \lambda_4 - 100.0 x_2 P_{1}ˏ_3 + 100.0 x_1 P_{2}ˏ_3 - 100.0 x_4 P_{3}ˏ_3 + 100.0 x_3 P_{4}ˏ_3 \right) - P_{1}ˏ_3 - P_{3}ˏ_1 - u_1 P_{3}ˏ_4 - u_1 P_{4}ˏ_3 \\
 - P_{4}ˏ_1 + \left(  - \lambda_3 + x_1 P_{2}ˏ_4 + x_3 P_{4}ˏ_4 - x_2 P_{1}ˏ_4 - x_4 P_{3}ˏ_4 \right) \left( 100.0 \lambda_4 - 100.0 x_2 P_{1}ˏ_3 + 100.0 x_1 P_{2}ˏ_3 - 100.0 x_4 P_{3}ˏ_3 + 100.0 x_3 P_{4}ˏ_3 \right) + u_1 P_{3}ˏ_3 - u_1 P_{4}ˏ_4 + P_{2}ˏ_3 \\
\left( x_1 P_{2}ˏ_1 + x_3 P_{4}ˏ_1 - x_2 P_{1}ˏ_1 - x_4 P_{3}ˏ_1 + \lambda_2 \right) \left(  - 100.0 \lambda_3 - 100.0 x_2 P_{1}ˏ_4 + 100.0 x_1 P_{2}ˏ_4 + 100.0 x_3 P_{4}ˏ_4 - 100.0 x_4 P_{3}ˏ_4 \right) + u_1 P_{1}ˏ_3 - u_1 P_{2}ˏ_4 + P_{1}ˏ_2 + P_{3}ˏ_4 \\
 - P_{4}ˏ_4 + \left(  - \lambda_1 + x_1 P_{2}ˏ_2 + x_3 P_{4}ˏ_2 - x_2 P_{1}ˏ_2 - x_4 P_{3}ˏ_2 \right) \left(  - 100.0 \lambda_3 - 100.0 x_2 P_{1}ˏ_4 + 100.0 x_1 P_{2}ˏ_4 + 100.0 x_3 P_{4}ˏ_4 - 100.0 x_4 P_{3}ˏ_4 \right) + u_1 P_{1}ˏ_4 + u_1 P_{2}ˏ_3 + P_{2}ˏ_2 \\
\left(  - 100.0 \lambda_3 - 100.0 x_2 P_{1}ˏ_4 + 100.0 x_1 P_{2}ˏ_4 + 100.0 x_3 P_{4}ˏ_4 - 100.0 x_4 P_{3}ˏ_4 \right) \left( x_1 P_{2}ˏ_3 + x_3 P_{4}ˏ_3 - x_2 P_{1}ˏ_3 - x_4 P_{3}ˏ_3 + \lambda_4 \right) - P_{1}ˏ_4 + u_1 P_{3}ˏ_3 - u_1 P_{4}ˏ_4 + P_{3}ˏ_2 \\
\left(  - \lambda_3 + x_1 P_{2}ˏ_4 + x_3 P_{4}ˏ_4 - x_2 P_{1}ˏ_4 - x_4 P_{3}ˏ_4 \right) \left(  - 100.0 \lambda_3 - 100.0 x_2 P_{1}ˏ_4 + 100.0 x_1 P_{2}ˏ_4 + 100.0 x_3 P_{4}ˏ_4 - 100.0 x_4 P_{3}ˏ_4 \right) + u_1 P_{3}ˏ_4 + u_1 P_{4}ˏ_3 + P_{2}ˏ_4 + P_{4}ˏ_2 \\
\end{array}
\right]
$$