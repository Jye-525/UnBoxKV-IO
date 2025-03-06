import json, time
import numpy as np
import threading
from queue import Queue

class AsyncJsonWriter:
    def __init__(self, kv_wr_output_path, kv_rd_output_path, max_queue_size=1000):
        self.kv_wr_output_path = kv_wr_output_path
        self.kv_rd_output_path = kv_rd_output_path
        self.max_queue_size = max_queue_size
        # write thread information
        self.kv_wr_queue = Queue()
        self.kv_wr_thread = threading.Thread(target=self._write)
        self.kv_wr_thread.start()

        # read thread information
        self.kv_rd_queue = Queue()
        self.kv_rd_thread = threading.Thread(target=self._read)
        self.kv_rd_thread.start() 

    def record_kv_write_pattern(self, data):
        self.kv_wr_queue.put(data)
        if self.kv_wr_queue.qsize() > self.max_queue_size:
            print(f'Warning:  KV write queue size is larger than max_queue_size', flush=True)

    def record_kv_read_pattern(self, data):
        self.kv_rd_queue.put(data)
        if self.kv_rd_queue.qsize() > self.max_queue_size:
            print(f'Warning:  KV read queue size is larger than max_queue_size', flush=True)

    def _write(self):
        with open(self.kv_wr_output_path, 'w') as f:
            while True:
                data = self.kv_wr_queue.get()
                if data is None:
                    break
                start = time.time()
                f.write(json.dumps(data) + '\n')
                # print(f'Write data cost (us): {(time.time()-start) * 1e6}', flush=True)
        print(f'AsyncJsonWriter KV write thread finished', flush=True)

    def _read(self):
        with open(self.kv_rd_output_path, 'w') as f:
            while True:
                data = self.kv_rd_queue.get()
                if data is None:
                    break
                start = time.time()
                f.write(json.dumps(data) + '\n')
                # print(f'Write data cost (us): {(time.time()-start) * 1e6}', flush=True)
        print(f'AsyncJsonWriter KV read thread finished', flush=True)

    def close(self):
        self.kv_wr_queue.put(None)
        time.sleep(1)
        self.kv_wr_thread.join()
        print(f'AsyncJsonWriter close KV write thread successfully.', flush=True)

        self.kv_rd_queue.put(None)
        time.sleep(1)
        self.kv_rd_thread.join()
        print(f'AsyncJsonWriter close KV read thread successfully.', flush=True) 

    def __del__(self):
        self.close()
