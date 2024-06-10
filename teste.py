import multiprocessing
import time

def worker(num):
    """Thread worker function"""
    print(f'Worker: {num}')
    time.sleep(1)
    return

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()