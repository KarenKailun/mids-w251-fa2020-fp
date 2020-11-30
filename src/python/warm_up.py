# Run this script at the end of the docker build process

# The rate detector download and processes some models
# this can be done as part of the docker build to
# speed-up load times.
from detector import BreathRateDetector

b = BreathRateDetector()