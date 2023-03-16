import psutil
from pprint import pprint
from pywinauto import Desktop
from pywinauto.application import Application
import win32gui
import win32process
import ctypes
from datetime import datetime
import time
import sys
import threading
import os

def get_parent_pid(process_name="tcw",completedSteps="tcflow"):

  pid = None
  DateAndTime = datetime.now()
  for proc in psutil.process_iter():
      if process_name in proc.name():
        pid = proc.pid
        break
      elif "Python" in proc.name():
        pid = proc.pid
        break
  # print ("pid",pid)
  if pid is not None:
    pp = psutil.Process(pid)
    # print('Parent: {:<20} pid: {:<20} status {:<20}'.format(
    #       pp.name(),pp.pid,pp.status()))
    # print('{:<20} | {:<20} | {:<20} | {:<20} | {:<20}'.format(
    #       "Parent","pid","status","memory_percent","cpu_percent"))
    # print("="*100)
    cpu = psutil.cpu_times_percent(interval=0.2)
    # print ("completedSteps",type(completedSteps),completedSteps)
    if completedSteps is None:
       c_step = "None"
    else:
      #  c_step = "\n".join(steps)
       steps = completedSteps
       if steps is None:
          c_step = "None"
       else:
          c_step = steps[-1]
    print('{:<20}|{:<10}|{:<10}|{:<10}|{:<10} | {:<20}    | {:<20}'.format(
          pp.name(),pp.pid,pp.status(),
          round(pp.memory_percent(),1),
          # pp.cpu_percent(interval=1),
          "%2.1f_%2.1f"%(cpu[0],cpu[1]),
          DateAndTime.strftime("%H:%M:%S-%D"),
          c_step
          ))

  return pid


GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
GetWindowText = ctypes.windll.user32.GetWindowTextW
IsWindowVisible = ctypes.windll.user32.IsWindowVisible
# Constants for Windows messages
WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E


def bring2front(hwnd):

  # Constants for the SetForegroundWindow function
  SW_SHOW = 5

  # Bring the message box to the front
  ctypes.windll.user32.SetForegroundWindow(hwnd)
  ctypes.windll.user32.ShowWindow(hwnd, SW_SHOW)


def get_hwnds_for_pid(pid):
    def callback(hwnd, hwnds):
        #if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)

        if found_pid == pid:
            hwnds.append(hwnd)
        return True
    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    return hwnds 

def getWindowTitleByHandle(hwnd):
    length = GetWindowTextLength(hwnd)
    buff = ctypes.create_unicode_buffer(length + 1)
    GetWindowText(hwnd, buff, length + 1)
    return buff.value


#######

def watchdog(completedSteps):
  
  # print ("watchdog **********************0")
  pid = get_parent_pid("tcw",completedSteps)
  # print ("watchdog **********************1")
  if pid is not None:
    current_process = psutil.Process(pid=pid)

    children = current_process.children(recursive=True)
    for child in children:
        print('C-{:<18}|{:<10}|{:<10}|{:<10}'.format(
          child.name(),
          child.pid,
          child.status(),
          round(child.memory_percent(),1)
          ))


  for hwnds in get_hwnds_for_pid(pid):
    ToCheck =["Error", "Not Found","Note"]
    ToSkip =["\n", 
              "DDE ",
              "IME",
              "Tools",
              "3D",
              "Drawing",
              "Standard",
              "Selection",
              "tcHelpWindow",
              "BroadcastEventWindow"]
    title = getWindowTitleByHandle(hwnds)
    if   any(Excpt in title for Excpt in ToSkip):
        pass
    elif  any(Excpt in title for Excpt in ToCheck):
        print ("warning: %s %s"%(title,IsWindowVisible(hwnds)))
        bring2front(hwnds)
        # print (hwnds)
        app = Application().connect(handle=hwnds)
        # app = Application().connect(process=pid)
        dlg_spec = app.window(title=title)
        ctridk = [str(k) for k in dlg_spec._ctrl_identifiers().keys()]
        
        # pprint (dlg_spec._ctrl_identifiers())
        if any("SysLink" in id for id in ctridk):
          print (dlg_spec.SysLink.texts()[0])
        
        elif any("Static" in id for id in ctridk):
          print (dlg_spec["Static"].texts()[0])
        else:
          pprint (dlg_spec._ctrl_identifiers())
        
        raise KeyboardInterrupt()

    else:
        pass
    # print (title)


from threading import Timer

# class RepeatTimer(Timer):
#     def run(self):
#         while not self.finished.wait(self.interval):
#             self.function(*self.args, **self.kwargs)


# def sub_thread(interval,modelRunTime):    
#     try:
#       while True:
#         watchdog(modelRunTime)
#         time.sleep(interval)
#         # global stop_threads
#         # print (stop_threads)
#         # if stop_threads:
#         #   break
#     except Exception:
#         print("Sub thread is interrupted")
#         raise KeyboardInterrupt()
#         sys.exit()

class sub_thread(threading.Thread):
    def __init__(self,completedSteps):
        threading.Thread.__init__(self)
        self.completedSteps = completedSteps
             
    def run(self):
 
        # target function of the thread class
        try:
            while True:
                watchdog(self.completedSteps)
                # print ("watchdog **********************3")
                time.sleep(30)
                # print ("watchdog **********************4")
                # print('running ' + self.name)
                if self.completedSteps == "emat":
                   break
        finally:
            os.system("TASKKILL /F /IM tcw.exe")
            print('Sub thread ended')
          
    def get_id(self):
 
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id
  
    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')
        

""" 
TASKKILL /F /IM tcw.exe
"""

if __name__ == "__main__":
   watchdog()