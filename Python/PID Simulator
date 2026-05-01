"""
PID Controller Simulator
========================
Simulates a closed-loop PID (Proportional-Integral-Derivative) controller
applied to a second-order system (e.g. motor speed or robot arm position).

The user can configure the target setpoint, PID gains, and simulation
parameters. The system response is plotted over time, showing key
performance metrics such as overshoot, rise time, and settling time.

Author: [egh50]
Unit:   EE22005 Engineering Practice and Design
Skill:  4.2 Python Programming (Synthesis) & 4.3 Software Design, Test, Validation
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Fix for VS Code — forces a display window to open
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# PIDController Class
# ---------------------------------------------------------------------------

class PIDController:
    """
    A discrete-time PID controller.

    Attributes
    ----------
    kp : float
        Proportional gain.
    ki : float
        Integral gain.
    kd : float
        Derivative gain.
    dt : float
        Time step (seconds).
    output_limits : tuple[float, float]
        (min, max) clamp on the controller output (anti-windup).
    """

    def __init__(self, kp: float, ki: float, kd: float, dt: float,
                 output_limits: tuple = (-float('inf'), float('inf'))):
        """
        Initialise the PID controller.

        Parameters
        ----------
        kp : float
            Proportional gain.
        ki : float
            Integral gain.
        kd : float
            Derivative gain.
        dt : float
            Simulation time step in seconds.
        output_limits : tuple, optional
            (min, max) output clamp. Defaults to no limit.
        """
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits

        # Internal state
        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def reset(self):
        """Reset internal integrator and previous error to zero."""
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, setpoint: float, measurement: float) -> float:
        """
        Compute the PID control output for one time step.

        Parameters
        ----------
        setpoint : float
            Desired target value.
        measurement : float
            Current measured system output.

        Returns
        -------
        float
            Control signal to apply to the plant.
        """
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term (with anti-windup via output clamping)
        self._integral += error * self.dt
        i_term = self.ki * self._integral

        # Derivative term (based on error change)
        derivative = (error - self._prev_error) / self.dt
        d_term = self.kd * derivative

        self._prev_error = error

        # Sum and clamp output
        output = p_term + i_term + d_term
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        return output


# ---------------------------------------------------------------------------
# SecondOrderPlant Class
# ---------------------------------------------------------------------------

class SecondOrderPlant:
    """
    A second-order linear plant model made of two first-order stages in series.

    Models a system such as a DC motor driving a mechanical load, where the
    electrical and mechanical dynamics each contribute a time constant.

    Discretised using the Euler forward method.

    Attributes
    ----------
    gain : float
        Static plant gain (K).
    tau1 : float
        Time constant of the first stage (seconds).
    tau2 : float
        Time constant of the second stage (seconds).
    dt : float
        Simulation time step.
    """

    def __init__(self, gain: float, tau1: float, tau2: float, dt: float):
        """
        Initialise the second-order plant.

        Parameters
        ----------
        gain : float
            Static gain of the plant.
        tau1 : float
            Time constant of the first stage (seconds). Must be positive.
        tau2 : float
            Time constant of the second stage (seconds). Must be positive.
        dt : float
            Simulation time step in seconds.
        """
        if tau1 <= 0 or tau2 <= 0:
            raise ValueError("Time constants must be positive.")
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")

        self.gain = gain
        self.tau1 = tau1
        self.tau2 = tau2
        self.dt = dt

        # Internal states for each stage
        self._x1: float = 0.0
        self._x2: float = 0.0

    @property
    def _output(self) -> float:
        """Return the current plant output (second stage)."""
        return self._x2

    def reset(self):
        """Reset both internal states to zero."""
        self._x1 = 0.0
        self._x2 = 0.0

    def step(self, control_input: float) -> float:
        """
        Advance the plant by one time step.

        Parameters
        ----------
        control_input : float
            Control signal from the PID controller.

        Returns
        -------
        float
            Updated plant output (second stage output).
        """
        # First stage: dx1/dt = (K*u - x1) / tau1
        dx1 = (self.gain * control_input - self._x1) / self.tau1
        self._x1 += dx1 * self.dt

        # Second stage: dx2/dt = (x1 - x2) / tau2
        dx2 = (self._x1 - self._x2) / self.tau2
        self._x2 += dx2 * self.dt

        return self._x2


# ---------------------------------------------------------------------------
# Simulation Function
# ---------------------------------------------------------------------------

def run_simulation(setpoint: float,
                   kp: float, ki: float, kd: float,
                   plant_gain: float = 1.0,
                   plant_tau1: float = 1.0,
                   plant_tau2: float = 0.5,
                   sim_time: float = 20.0,
                   dt: float = 0.01,
                   output_limits: tuple = (-50.0, 50.0)) -> dict:
    """
    Run a closed-loop PID simulation and return time-series results.

    Parameters
    ----------
    setpoint : float
        Target value the system should reach and maintain.
    kp : float
        Proportional gain.
    ki : float
        Integral gain.
    kd : float
        Derivative gain.
    plant_gain : float
        Static gain of the plant model.
    plant_tau1 : float
        First time constant of the plant model (seconds).
    plant_tau2 : float
        Second time constant of the plant model (seconds).
    sim_time : float
        Total simulation duration (seconds).
    dt : float
        Time step size (seconds).
    output_limits : tuple
        (min, max) clamp on controller output.

    Returns
    -------
    dict
        Dictionary with keys: 'time', 'output', 'error', 'control',
        and 'metrics' (rise_time, overshoot_pct, settling_time).
    """
    if setpoint == 0:
        raise ValueError("Setpoint must be non-zero to compute meaningful metrics.")

    controller = PIDController(kp, ki, kd, dt, output_limits)
    plant = SecondOrderPlant(plant_gain, plant_tau1, plant_tau2, dt)

    steps = int(sim_time / dt)
    time = np.linspace(0, sim_time, steps)
    output = np.zeros(steps)
    error = np.zeros(steps)
    control = np.zeros(steps)

    for i in range(steps):
        measurement = plant._output
        u = controller.compute(setpoint, measurement)
        plant.step(u)

        output[i] = measurement
        error[i] = setpoint - measurement
        control[i] = u

    metrics = compute_metrics(time, output, setpoint)

    return {
        'time': time,
        'output': output,
        'error': error,
        'control': control,
        'metrics': metrics
    }


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics(time: np.ndarray, output: np.ndarray,
                    setpoint: float) -> dict:
    """
    Compute standard step-response performance metrics.

    Parameters
    ----------
    time : np.ndarray
        Time array.
    output : np.ndarray
        System output array.
    setpoint : float
        Target setpoint value.

    Returns
    -------
    dict
        Dictionary with 'rise_time', 'overshoot_pct', 'settling_time'.
        Values are None if the metric cannot be determined.
    """
    metrics = {
        'rise_time': None,
        'overshoot_pct': None,
        'settling_time': None
    }

    # Rise time: time to go from 10% to 90% of setpoint
    ten_pct = 0.10 * setpoint
    ninety_pct = 0.90 * setpoint
    t10 = next((time[i] for i in range(len(output)) if output[i] >= ten_pct), None)
    t90 = next((time[i] for i in range(len(output)) if output[i] >= ninety_pct), None)
    if t10 is not None and t90 is not None:
        metrics['rise_time'] = round(t90 - t10, 3)

    # Overshoot: maximum value beyond setpoint
    peak = np.max(output)
    if peak > setpoint:
        metrics['overshoot_pct'] = round(((peak - setpoint) / setpoint) * 100, 2)
    else:
        metrics['overshoot_pct'] = 0.0

    # Settling time: last time output leaves ±2% band
    band = 0.02 * abs(setpoint)
    settled = np.where(np.abs(output - setpoint) > band)[0]
    if len(settled) > 0:
        metrics['settling_time'] = round(time[settled[-1]], 3)
    else:
        metrics['settling_time'] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Plotting Function
# ---------------------------------------------------------------------------

def plot_results(results_list: list, labels: list, setpoint: float):
    """
    Plot simulation results for one or more PID configurations.

    Parameters
    ----------
    results_list : list[dict]
        List of result dictionaries from run_simulation().
    labels : list[str]
        Labels for each configuration (for the legend).
    setpoint : float
        The target setpoint value.
    """
    colours = ['#FF5722', '#4CAF50', '#2196F3']

    fig = plt.figure(figsize=(14, 9), facecolor='#0d1117')
    fig.suptitle('PID Controller Simulator — Step Response Analysis | egh50',
                 fontsize=15, color='white', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_output  = fig.add_subplot(gs[0, :])   # Full width top
    ax_error   = fig.add_subplot(gs[1, 0])
    ax_control = fig.add_subplot(gs[1, 1])

    for ax in [ax_output, ax_error, ax_control]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # --- Output plot ---
    ax_output.axhline(setpoint, color='#ffd700', linewidth=1.2,
                      linestyle='--', label=f'Setpoint = {setpoint}')
    ax_output.axhspan(setpoint * 0.98, setpoint * 1.02,
                      alpha=0.08, color='#ffd700', label='±2% settling band')

    for i, (res, label) in enumerate(zip(results_list, labels)):
        c = colours[i % len(colours)]
        ax_output.plot(res['time'], res['output'], color=c,
                       linewidth=1.8, label=label)
        m = res['metrics']
        info = (f"  Rise: {m['rise_time']}s | "
                f"Overshoot: {m['overshoot_pct']}% | "
                f"Settling: {m['settling_time']}s")
        ax_output.annotate(label + info, xy=(0.01, 0.97 - i * 0.07),
                           xycoords='axes fraction', fontsize=7.5,
                           color=c, va='top')

    ax_output.set_title('System Output vs Setpoint')
    ax_output.set_xlabel('Time (s)')
    ax_output.set_ylabel('Output')
    ax_output.legend(fontsize=8, facecolor='#161b22',
                     labelcolor='white', edgecolor='#30363d')
    ax_output.grid(True, color='#30363d', linewidth=0.5)

    # --- Error plot ---
    for i, (res, label) in enumerate(zip(results_list, labels)):
        ax_error.plot(res['time'], res['error'],
                      color=colours[i % len(colours)],
                      linewidth=1.5, label=label)
    ax_error.axhline(0, color='white', linewidth=0.6, linestyle=':')
    ax_error.set_title('Tracking Error Over Time')
    ax_error.set_xlabel('Time (s)')
    ax_error.set_ylabel('Error')
    ax_error.legend(fontsize=8, facecolor='#161b22',
                    labelcolor='white', edgecolor='#30363d')
    ax_error.grid(True, color='#30363d', linewidth=0.5)

    # --- Control signal plot ---
    for i, (res, label) in enumerate(zip(results_list, labels)):
        ax_control.plot(res['time'], res['control'],
                        color=colours[i % len(colours)],
                        linewidth=1.5, label=label)
    ax_control.set_title('Control Signal (Actuator Input)')
    ax_control.set_xlabel('Time (s)')
    ax_control.set_ylabel('Control Output')
    ax_control.legend(fontsize=8, facecolor='#161b22',
                      labelcolor='white', edgecolor='#30363d')
    ax_control.grid(True, color='#30363d', linewidth=0.5)

    plt.savefig('pid_simulation_results.png', dpi=150,
                bbox_inches='tight', facecolor='#0d1117')
    plt.show()
    print("\nPlot saved as 'pid_simulation_results.png'")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """
    Main function: defines simulation configurations and runs the simulator.

    Three PID tunings are compared side-by-side to illustrate the effect
    of gain selection on system response — a common robotics design task.
    """
    SETPOINT   = 1.0   # Target output (e.g. 1 m/s wheel speed or 1 rad position)
    SIM_TIME   = 20.0  # Simulation duration (seconds)
    DT         = 0.01  # Time step (seconds)

    # Plant parameters (second-order model)
    PLANT_GAIN = 1.0
    PLANT_TAU1 = 1.0   # First time constant (e.g. electrical dynamics)
    PLANT_TAU2 = 0.5   # Second time constant (e.g. mechanical dynamics)

    # Three different PID tunings to compare
    configs = [
        {'label': 'Under-damped  (Kp=8,  Ki=5,  Kd=0.1)', 'kp': 8.0, 'ki': 5.0, 'kd': 0.1},
        {'label': 'Well-tuned    (Kp=3,  Ki=1,  Kd=1.5)', 'kp': 3.0, 'ki': 1.0, 'kd': 1.5},
        {'label': 'Over-damped   (Kp=0.5,Ki=0.2,Kd=0.1)', 'kp': 0.5, 'ki': 0.2, 'kd': 0.1},
    ]

    results = []
    labels  = []

    print("=" * 60)
    print("  PID Controller Simulator")
    print("=" * 60)
    print(f"  Setpoint : {SETPOINT}")
    print(f"  Plant    : Gain={PLANT_GAIN}, Tau1={PLANT_TAU1}s, Tau2={PLANT_TAU2}s")
    print(f"  Duration : {SIM_TIME}s  |  dt={DT}s")
    print("=" * 60)

    for cfg in configs:
        res = run_simulation(
            setpoint=SETPOINT,
            kp=cfg['kp'], ki=cfg['ki'], kd=cfg['kd'],
            plant_gain=PLANT_GAIN,
            plant_tau1=PLANT_TAU1,
            plant_tau2=PLANT_TAU2,
            sim_time=SIM_TIME, dt=DT,
            output_limits=(-50.0, 50.0)
        )
        results.append(res)
        labels.append(cfg['label'])

        m = res['metrics']
        print(f"\n{cfg['label']}")
        print(f"  Rise time    : {m['rise_time']} s")
        print(f"  Overshoot    : {m['overshoot_pct']} %")
        print(f"  Settling time: {m['settling_time']} s")

    print("\n" + "=" * 60)
    print("  Generating plot...")
    plot_results(results, labels, SETPOINT)


if __name__ == '__main__':
    main()
