import numpy as np
import matplotlib.pyplot as plt


def rf_pulse(duration, dt):
    return np.sin(np.pi * np.linspace(-2*np.pi, 2*np.pi, int(duration/dt)))/ (np.pi * np.linspace(-2*np.pi, 2*np.pi, int(duration/dt)))




TE = 15
Delta = 10
Start = 1.25
rf_chan_offset = 1.5
rf_pulse_duration = 1.0
ts, dt = np.linspace(0, TE, 2048, retstep=True )
delta = dt
pulse_width = 1*int(dt/dt)

rf_channel = np.zeros(ts.shape) + rf_chan_offset
gradient_channel = np.zeros(ts.shape)
echo_channel = np.zeros(ts.shape) - rf_chan_offset



rf_channel[0:int(rf_pulse_duration/dt)] = rf_pulse(rf_pulse_duration, dt) + rf_chan_offset
rf_channel[int(rf_pulse_duration/dt) + int(Delta/(2*dt)):  int(Delta/(2*dt)) + 2*int(rf_pulse_duration/dt)] = rf_pulse(rf_pulse_duration, dt) + rf_chan_offset

gradient_channel[int(Start/dt): int(Start/dt) + pulse_width] = 1.0
gradient_channel[int(Start/dt) + int(Delta/dt): int(Start/dt) + int(Delta/dt) + pulse_width] = 1.0


echo_channel[int(TE/dt) - int(3/dt) + 1 : ] = np.exp(-0.5 * np.linspace(0, 4, int(3/dt))) * np.sin(5*np.pi * np.linspace(0, 2*np.pi, int(3/dt))) - rf_chan_offset




fig, ax = plt.subplots(figsize = (10,3))

ax.plot(ts, rf_channel, color = 'black')
ax.plot(ts, gradient_channel, color = 'black')
ax.plot(ts, echo_channel, color = 'black')
ax.axis('off')
plt.show()
