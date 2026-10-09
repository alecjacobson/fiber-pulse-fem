"""Post-run physical measurements; no host analysis occurs during propagation."""
import numpy as np
from scipy.signal import hilbert
from fiber_pulse import C0, Z0


def metrics(t,trace,ports,mode,ng,center,run,modes=None,norms=None):
    all_trace = trace
    trace = trace[:,:len(ports)]
    dt = t[1]-t[0]
    omega_fs = 2*np.pi*C0/run.wavelength_um*1e-9
    admittance = mode['neff']
    forward = 0.5*(trace[:,:,0]-trace[:,:,1]/admittance)
    backward = 0.5*(trace[:,:,0]+trace[:,:,1]/admittance)
    # Phasors with e^(+i wt) convention; right-going signal has phase -beta*x.
    phasors = np.sum(forward*np.exp(-1j*omega_fs*t[:,None]),axis=0)*dt
    # Adjacent probes are < guided wavelength; unwrap using the analytic beta
    # as a branch anchor, then fit the numerical beta without fixing its slope.
    x = ports[:-1]
    residual = np.unwrap(np.angle(phasors[:-1]*np.exp(1j*mode['beta']*x)))
    slope,intercept = np.polyfit(x,residual,1)
    beta = mode['beta']-slope
    residual_rms = np.sqrt(np.mean((residual-(slope*x+intercept))**2))
    env2 = np.abs(hilbert(forward,axis=0))**2
    arrivals = np.sum(t[:,None]*env2,axis=0)/np.sum(env2,axis=0)
    # Main input/output monitors: x=4.6 and x=14.5 um.
    ip,op = 3,len(ports)-1
    ein = np.trapz(forward[:,ip]**2,t)
    eout = np.trapz(forward[:,op]**2,t)
    back = np.trapz(backward[:,ip]**2,t)
    path = np.linalg.norm(np.diff(center,axis=0),axis=1)
    s = np.r_[0,np.cumsum(path)]
    length = np.interp(ports[op],center[:,0],s)-np.interp(ports[ip],center[:,0],s)
    expected_delay = length*ng*1e9/C0
    delay = arrivals[op]-arrivals[ip]
    flux = np.trapz(trace[:,:,2],t,axis=0)*1e-15*1e-6*1e12/Z0
    guided_flux = np.trapz(trace[:,:,3],t,axis=0)*1e-15*1e-6*1e12/Z0
    power = trace[:,:,2]*1e-6*1e12/Z0
    result = dict(beta_per_um=float(beta),beta_relative_error=float((beta/mode['beta'])-1),
                beta_phase_fit_rms_rad=float(residual_rms),
                modal_transmission=float(eout/ein),modal_loss_db=float(-10*np.log10(eout/ein)),
                carrier_modal_transmission=float(abs(phasors[op]/phasors[ip])**2),
                backward_to_forward_energy=float(back/ein),
                net_flux_transmission=float(flux[op]/flux[ip]),
                window_flux_transmission=float(guided_flux[op]/guided_flux[ip]),
                input_fluence_J_per_m=float(flux[ip]),output_fluence_J_per_m=float(flux[op]),
                peak_input_power_W_per_m=float(np.max(power[:,ip])),
                peak_output_power_W_per_m=float(np.max(power[:,op])),
                input_arrival_fs=float(arrivals[ip]),output_arrival_fs=float(arrivals[op]),
                delay_fs=float(delay),geometric_delay_fs=float(expected_delay),
                delay_relative_error=float(delay/expected_delay-1),path_length_um=float(length),
                straight_group_index_measured=float(np.polyfit(x,arrivals[:-1],1)[0]*C0/1e9),
                tail_fraction=float(np.sum(env2[-max(1,len(t)//10):,op])/np.sum(env2[:,op])))
    if modes is not None:
        modal_out = [float(eout/ein)]
        modal_in = [1.0]
        for i,md in enumerate(modes[1:]):
            j = len(ports)+2*i
            a = 0.5*(all_trace[:,j:j+2,0]-all_trace[:,j:j+2,1]/md['neff'])
            energy = np.trapz(a*a,t,axis=0)*md['neff']*norms[j:j+2]/(ein*mode['neff']*norms[ip])
            modal_in.append(float(energy[0]))
            modal_out.append(float(energy[1]))
        result.update(mode_transmissions=modal_out,input_mode_energy_ratios=modal_in,
                      total_guided_transmission=float(sum(modal_out)))
    return result
