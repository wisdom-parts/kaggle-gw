from gw_data import SIGNAL_SECS, N_SIGNALS

# We separate these constants from the preprocessing module because it goes badly for
# wandb's background process to try to import pycbc.

FREQ_STEPS = 32
TIME_STEPS_PER_SEC = 64
TIME_STEPS = round(SIGNAL_SECS * TIME_STEPS_PER_SEC)
OUTPUT_SHAPE = (N_SIGNALS, FREQ_STEPS, TIME_STEPS)
FREQ_STEPS_64x256 = 64
TIME_STEPS_PER_SEC_64x256 = 128
TIME_STEPS_64x256 = round(SIGNAL_SECS * TIME_STEPS_PER_SEC_64x256)
OUTPUT_SHAPE_64x256 = (N_SIGNALS, FREQ_STEPS_64x256, TIME_STEPS_64x256)