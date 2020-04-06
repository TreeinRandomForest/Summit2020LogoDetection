package main

accuracyThreshold := 1.0

deny[msg] {
    input.accuracy < accuracyThreshold
    msg := sprintf("Submitted model accuracy value %v%% is unacceptable. Model accuracy must be greater than or equal to %v%%.", [input.accuracy * 100, accuracyThreshold * 100])
}
