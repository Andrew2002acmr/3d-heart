"""Measured process/system resources for a bounded CPU experiment."""
import platform
import threading
import time
import psutil
import torch


def environment(device='cpu'):
    processor=platform.processor()
    if platform.system()=='Windows':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            processor=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    return {'platform':platform.platform(),'python':platform.python_version(),
        'processor':processor,'logical_CPUs':psutil.cpu_count(),
        'physical_CPUs':psutil.cpu_count(logical=False),'RAM_bytes':psutil.virtual_memory().total,
        'torch':torch.__version__,'torch_threads':torch.get_num_threads(),
        'CUDA_available':torch.cuda.is_available(),'benchmark_device':device}


class ResourceMonitor:
    def __init__(self,interval=.2):
        self.interval=interval;self.samples=[];self.stop_event=threading.Event()
        self.process=psutil.Process()

    def _sample(self):
        self.process.cpu_percent(None)
        while True:
            memory=psutil.virtual_memory()
            self.samples.append({'time_monotonic':time.perf_counter(),
                'process_RSS_bytes':self.process.memory_info().rss,
                'process_CPU_percent':self.process.cpu_percent(None),
                'system_available_RAM_bytes':memory.available,'system_RAM_percent':memory.percent})
            if self.stop_event.wait(self.interval):break

    def __enter__(self):
        self.thread=threading.Thread(target=self._sample,daemon=True);self.thread.start();return self

    def __exit__(self,*args):
        self.stop_event.set();self.thread.join(timeout=2)

    def summary(self):
        if not self.samples:raise ValueError('No resource measurements')
        cpu=sum(s['process_CPU_percent'] for s in self.samples)/len(self.samples)
        return {'samples':len(self.samples),'sampling_interval_seconds':self.interval,
            'peak_process_RSS_bytes':max(s['process_RSS_bytes'] for s in self.samples),
            'mean_process_CPU_percent':cpu,'mean_process_CPU_percent_of_logical_machine':cpu/psutil.cpu_count(),
            'minimum_system_available_RAM_bytes':min(s['system_available_RAM_bytes'] for s in self.samples),
            'maximum_system_RAM_percent':max(s['system_RAM_percent'] for s in self.samples),
            'CPU_percent_definition':'100% process = one logical CPU; all process threads included',
            'memory_definition':'sampled working-set RSS; excludes OS file cache, includes Python/torch/cache pages'}
