import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import exp

# Create plotting functions

def plotsir(t, S, U, R, L = None):
  plt.rcParams["font.family"] = "Baskerville"
  f, ax = plt.subplots(1,1,figsize=(10,4))
  ax.plot(t, S, 'c', alpha=0.7, linewidth=2, label='Susceptible')
  ax.plot(t, U, 'm', alpha=0.7, linewidth=2, label='Active')
  ax.plot(t, R, 'y', alpha=0.7, linewidth=2, label='Recovered')
  ax.set_xlabel('Time (t)')
  ax.set_ylabel('Number of Individuals')

  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)
  ax.grid(b=True, which='major', c='w', lw=2, ls='-')
  legend = ax.legend(borderpad=2.0)
  legend.get_frame().set_alpha(0.5)
  for spine in ('right','left'):
      ax.spines[spine].set_visible(True)
  if L is not None:
      plt.title("Level of Unrest after {} Time Units".format(L)) # What time scale are we looking at?
  plt.show();

# Define function to plot the Infected solutions for compartments

def plotinfec(t, U1, U2, U3, U4, U5, L=None):

    plt.rcParams["font.family"] = "Baskerville"
    plt.rcParams["font.size"] = 13
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, U1, 'r', alpha=0.7, linewidth=2, label='Active, Compartment 1')
    ax.plot(t, U2, 'm', alpha=0.7, linewidth=2, label='Active, Compartment 2')

    if U3 is not None:
        ax.plot(t, U3, 'c', alpha=0.7, linewidth=2, label='Active, Compartment 3')
    if U4 is not None:
        ax.plot(t, U4, 'y', alpha=0.7, linewidth=2, label='Active, Compartment 4')
    if U5 is not None:
        ax.plot(t, U5, 'b', alpha=0.7, linewidth=2, label='Active, Compartment 5')

    hfont = {'fontname': 'Baskerville',
             'size': 14,
             'weight': 'bold'}

    ax.set_xlabel('Time (t)', **hfont)
    ax.set_ylabel('Number of Individuals', **hfont)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend(borderpad=2.0)
    legend.get_frame().set_alpha(0.5)
    for spine in ('right', 'left'):
        ax.spines[spine].set_visible(True)
    if L is not None:
        plt.title("Level of Unrest after {} Time Units".format(L))  # What time scale are we looking at?
    plt.show();

# Define matrix of compartment connections
# Eventually the model will be extended to nine compartments which is why W is 9x9

w01, w02, w03, w04, w05, w06, w07, w08 = 0, 0, 0, 0, 0, 0, 0, 0
w10, w12, w13, w14, w15, w16, w17, w18 = 10e-4, 0, 0, 0, 0, 0, 0, 0
w20, w21, w23, w24, w25, w26, w27, w28 = 0, 0, 0, 0, 0, 0, 0, 0
w30, w31, w32, w34, w35, w36, w37, w38 = 0, 0, 0, 0, 0, 0, 0, 0
w40, w41, w42, w43, w45, w46, w47, w48 = 0, 0, 0, 0, 0, 0, 0, 0
w50, w51, w52, w53, w54, w56, w57, w58 = 0, 0, 0, 0, 0, 0, 0, 0
w60, w61, w62, w63, w64, w65, w67, w68 = 0, 0, 0, 0, 0, 0, 0, 0
w70, w71, w72, w73, w74, w75, w76, w78 = 0, 0, 0, 0, 0, 0, 0, 0
w80, w81, w82, w83, w84, w85, w86, w87 = 0, 0, 0, 0, 0, 0, 0, 0

# Matrix with the interaction terms

w = np.array([[0, w01, w02, w03, w04, w05, w06, w07, w08],
              [w10, 0, w12, w13, w14, w15, w16, w17, w18],
              [w20, w21, 0, w23, w24, w25, w26, w27, w28],
              [w30, w31, w32, 0, w34, w35, w36, w37, w38],
              [w40, w41, w42, w43, 0, w45, w46, w47, w48],
              [w50, w51, w52, w53, w54, 0, w56, w57, w58],
              [w60, w61, w62, w63, w64, w65, 0, w67, w68],
              [w70, w71, w72, w73, w74, w75, w76, 0, w78],
              [w80, w81, w82, w83, w84, w85, w86, w87, 0]])

# Define function with SIR equations

def fnc1(y, t, beta, gamma, delta, w, tau):
    S1, U1, R1, S2, U2, R2, S3, U3, R3, S4, U4, R4 = y

    S = np.array([S1, S2, S3, S4])
    U = np.array([U1, U2, U3, U4])

    dS1dt = -beta * S1 * U1 + delta * R1
    dU1dt = beta * S1 * U1 - gamma * U1
    dR1dt = gamma * U1 - delta * R1

    dS2dt = - beta * S2 * U2 + delta * R2
    dU2dt = beta * S2 * U2 - gamma * U2
    dR2dt = gamma * U2 - delta * R2

    dS3dt = -beta * S3 * U3 + delta * R3
    dU3dt = beta * S3 * U3 - gamma * U3
    dR3dt = gamma * U3 - delta * R3

    dS4dt = -beta * S4 * U4 + delta * R4
    dU4dt = beta * S4 * U4 - gamma * U4
    dR4dt = gamma * U4 - delta * R4

    if t > tau:
        for j in range(3):
            dU1dt += w[0, j] * U[j] * S[0]
            dS1dt -= w[0, j] * U[j] * S[0]
            dU2dt += w[1, j] * U[j] * S[1]
            dS2dt -= w[1, j] * U[j] * S[1]
            dU3dt += w[2, j] * U[j] * S[2]
            dS3dt -= w[2, j] * U[j] * S[2]
            dU4dt += w[3, j] * U[j] * S[3]
            dS4dt -= w[3, j] * U[j] * S[3]

    return [dS1dt, dU1dt, dR1dt, dS2dt, dU2dt, dR2dt, dS3dt, dU3dt, dR3dt, dS4dt, dU4dt, dR4dt]

# Set constant values and initial conditions

N1 = 100
N2 = 100
U1 = 1 # Start with one infected (active) person in Comp 1
U2 = 0 # No active unrest in Compartment 2
U3 = 0
U4 = 0

R1 = R2 = R3 = R4 = 0 # Initially, there are no recovered persons
S1 = N1 - U1 - R1
S2 = N2 - U2 - R2
S3 = N1 - U3 - R3
S4 = N1 - U4 - R4

# Transmission, recovery (with immunity) and re-susceptibility parameter

beta, gamma, delta = 1/5, 1/7, 10e-4

# Vector of initial conditions
Y0 = np.array([S1, U1, R1, S2, U2, R2, S3, U3, R3, S4, U4, R4])
t = np.linspace(0, 60, 100)

# Time delay parameter
tau = 0

sol = odeint(fnc1, Y0, t, args=(beta, gamma, delta, w, tau))

# Extract individual solutions

Susceptible_1 = sol[:, 0]
Active_1 = sol[:, 1]
Recovered_1 = sol[:, 2]
Susceptible_2 = sol[:, 3]
Active_2 = sol[:, 4]
Recovered_2 = sol[:, 5]
Susceptible_3 = sol[:, 6]
Active_3 = sol[:, 7]
Recovered_3 = sol[:, 8]

# Plot solutions
plotinfec(t, Active_1, Susceptible_1, Active_2, Recovered_2, None)
