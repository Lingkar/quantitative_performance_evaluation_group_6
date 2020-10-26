import os
import gc


def saveToFile(labeler, ks, label_error, image_noise, epochs, count, accuracy, replication_set):
    with open("intermediate_results.txt", "a") as result_file:
        result_file.write("\n")
        result_file.write(
            "Replication_set: " + str(replication_set) + "Index: " + str(count) + ", labeler: " + str(labeler) + ", ks: " + str(ks) + ", label_error: " + str(
                label_error) + ", image_noise: " + str(image_noise) + ", epochs: " + str(epochs) + "\n")
        result_file.write(str(accuracy))


if __name__ == "__main__":
    replication_set = 3
    labeler = [0, 1]
    kernel_size = [3]
    experiments = [[40, 40, 20], [40, 0, 20], [0, 40, 20], [0, 0, 20]]
    thr = 0.1
    alpha = 1.5
    beta = 1
    result_full_matrix = []
    count = 0
    for i in labeler:
        for j in kernel_size:
            for e in experiments:
                if count > -1:
                    label_error = e[0]
                    image_noise = e[1]
                    epochs = e[2]
                    command = 'python3 main.py -s mnist -a ' + str(alpha) + ' -b ' + str(beta) + ' -t ' + str(
                        thr) + ' -ks1 ' + str(j) + ' -ks2 ' + str(j) + ' -e ' + str(epochs) + ' -er ' + str(
                        label_error) + ' -in ' + str(image_noise) + ' -la ' + str(i) + ' -re ' + str(replication_set)
                    print("INDEX: " + str(count))
                    print(command)
                    stream = os.popen(command)
                    output = stream.read().split()
                    # print(output)
                    accuracy = output[len(output) - 1]
                    # result_repetitions.append(accuracy)
                    # print(result_repetitions)
                    # if r == repetition - 1:
                    saveToFile(i, j, label_error, image_noise, epochs, count, accuracy, replication_set)
                count = count + 1
    print("Done!")
