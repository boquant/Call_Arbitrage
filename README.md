# Call_Arbitrage

This project implements a local volatility pricing framework for European/American (no dividend) call. The pipeline includes:

Extracting implied volatility from market data

Building the IV and call price surfaces via interpolation

Computing local volatility using Dupire’s formula

Solving the PDE using the Crank–Nicolson method

Limitations
Instability: The local volatility surface is highly sensitive to noise in the call surface, especially in the second derivative, often leading to blow-ups or NaNs in the final price.

Manual domain bounds: The finite difference grid requires subjective x_min and x_max, which introduces arbitrary risk assumptions and breaks the no-arbitrage principle.

Future Improvements
Replace local volatility with more stable models like Heston or LSV that better capture skew and avoid numerical issues.

Use dynamic or model-consistent criteria (e.g. delta or IV quantiles) to set PDE grid boundaries instead of manual guesses.

