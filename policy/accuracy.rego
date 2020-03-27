package main

accuracyThreshold := 1.0

deny[msg] {
    input.accuracy < accuracyThreshold
    msg := sprintf("Unacceptable accuracy value %v%%. Accuracy must be greater than or equal to %v%%.", [input.accuracy * 100, accuracyThreshold * 100])
}
