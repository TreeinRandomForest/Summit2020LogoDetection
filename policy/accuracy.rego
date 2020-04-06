package main

accuracyThreshold := 1.0

deny[msg] {
    accuracy := to_number(input.accuracy)
    accuracy < accuracyThreshold
    msg := sprintf("Submitted model accuracy value %v%% is unacceptable. Model accuracy must be greater than or equal to %v%%.", [accuracy * 100, accuracyThreshold * 100])
}
