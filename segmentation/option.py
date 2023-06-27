import matplotlib.pyplot as plt

def draw_process(learning_log, path):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(learning_log['epoch'], learning_log['loss'], label='train_loss')
    ax1.plot(learning_log['epoch'], learning_log['val_loss'], label='val_loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend(loc='best')
    ax1.grid(True)
    ax1.set_title('Loss')
    plt.savefig(path + 'Loss.png')

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    ax2.plot(learning_log['epoch'], learning_log['iou'], label='train_iou')
    ax2.plot(learning_log['epoch'], learning_log['val_iou'], label='val_iou')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('IoU')
    ax2.legend(loc='best')
    ax2.grid(True)
    ax2.set_title('IoU')
    plt.savefig(path + 'Accuracy.png')
