package main

accuracyThreshold := 0.8

deny[msg] {
    input.accuracy < accuracyThreshold
    msg := sprintf("Unacceptable accuracy value %v. Accuracy must be greater than or equal to %v.", [input.accuracy, accuracyThreshold])
}
