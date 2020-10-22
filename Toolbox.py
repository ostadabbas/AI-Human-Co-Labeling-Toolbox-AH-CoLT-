from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from tkinter import messagebox
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib
plt.switch_backend('TkAgg')
import copy
import glob

import helpers
import AI_models

import sys
x=5000
sys.setrecursionlimit(x)

LARGE_FONT = ("Verdana", 12)  # font's family is Verdana, font's size is 12

# dictionary of keypoints annotation
dict_model = {
        "Hourglass":" 0: R_Ankle, 1: R_Knee, 2: R_Hip, 3: L_Hip, 4: L_Knee, 5: L_Ankle, 6: Pelv, 7: Thrx, 8: Neck, 9: Head,"
                    " 10: R_Wrist, 11: R_Elbow, 12: R_Shoulder, 13: L_Shoulder, 14: L_Elbow, 15: L_Wrist",

        "Faster R-CNN":" 0: Nose, 1: L_Eye, 2: R_Eye, 3: L_Ear, 4: R_Ear, 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow,"
                     " 8: R_Elbow, 9: L_Wrist, 10: R_Wrist, 11: L_Hip, 12: R_Hip, 13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle",

        "FAN":" 1-17: Jaw, 18-22: L_Eyebrow, 23-27: R_Eyebrow, 28-36: Nose, 37-42: L_Eye, 43-48: R_Eye, "               
             " 49-60: Outer_Lip, 61-68: Inner_Lip",

        }

# frame geometry dict
geometry = {
           "BranchMenu": "700x300",
           "BodyMenu": "700x300", 
           "FaceMenu": "700x300",
           "AI_Labeler": "600x320",
           "Human_Reviewer": "620x200",
           "Human_Reviser": "620x200"
           }

# global variables
target = os.getcwd()  # save results under root path
branch = 0 # branch index 0: none, 1: body, 2: face
num_kpts = 0  # number of keypoints of current pose
num_poses = 0  # number of poses of current image
txt_list = []  # record texts on current canvas
pt_list = []  # record points on current canvas

dict_flags = {}  # save flags for image set
flag = []  # record flags ('0': reject, '1': accept, '-1': delete, '2': insert) of current image

fix = []  # record indexes of keypoints, which need be corrected, for current image.
fixed = []  # save revised x, y coordinate and visibility of  keypoints for current image
vis_idx = []  # record indexes of keypoints to display in canvas 
vis_pose_idx = []  # record indexes of poses to display in canvas
result = {}  # save generated groundtruth  for image set
bbox = []  # save head bounding box (x1, y1, x2, y2) of current image


class MainWindow(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        self.title("AI Human Co-labeling Toolbox (AH-CoLT)")  # set the title of the main window
        self.geometry("580x340")  # set size of the main window to 580X340 pixels
        self.resizable(False, False)

        # this container contains all the pages
        self.container = Frame(self)
        self.container.pack(side= "top", fill = "both", expand = True)
        self.container.grid_rowconfigure(0, weight=1)  # make the cell in grid cover the entire window
        self.container.grid_columnconfigure(0, weight=1)  # make the cell in grid cover the entire window
        '''
        self.frames = {}  # these are pages we want to navigate to
        # Style().configure("My.TFrame", background='#fff4f7')
        for F,geometry in zip((BranchMenu, BodyMenu, FaceMenu, AI_Labeler, Human_Reviewer, Human_Reviser), ("700x300", "700x300", "700x300", "600x320", "620x200", "620x200")):  # for each page
            frame = F(container, self)  # create the page
            self.frames[F] = (frame, geometry)  # store into frames
            # frame.config(style='My.TFrame')
            frame.grid(row=0, column=0, sticky="nsew")  # grid it to container
        '''
        self.show_frame(BranchMenu, "BranchMenu", 0)  # let the first page is StartPage

    def show_frame(self, F, F_name, branch_idx):
        global branch
        branch = branch_idx

        frame = F(self.container, self)
        self.geometry(geometry[F_name])
        frame.grid(row=0, column=0, sticky="nsew")  # grid it to container
        frame.tkraise()

    def exit(self):
        self.destroy()

class BranchMenu(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        # Style().configure("My.TLabel", background='#fff4f7')
        title = Label(self, text="AI Human Co-labeling Toolbox (AH-CoLT)", font="none 20 bold") #none 20 bold
        title.pack(pady=10, padx=10)  # center alignment

        Style().configure("My.TButton", font=('Helvetica', 12))
        # Style().map('My.TButton', background=[('active', 'red')])
        button1 = Button(self, text='Body Keypoints Annotation', command=lambda: controller.show_frame(BodyMenu, "BodyMenu", 1), style='My.TButton')
        button1.pack(fill="both", pady=10, padx=200, expand=True)
        button2 = Button(self, text='Faical Landmarks Annotation', command=lambda: controller.show_frame(FaceMenu, "FaceMenu", 2), style='My.TButton')
        button2.pack(fill="both", pady=10, padx=200, expand=True)
        button4 = Button(self, text='Exit', command=lambda: controller.exit(), style='My.TButton')
        button4.pack(fill="both", pady=10, padx=200, expand=True)


class BodyMenu(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        # Style().configure("My.TLabel", background='#fff4f7')
        title = Label(self, text="Body Keypoints Annotation", font="none 20 bold") #none 20 bold
        title.pack(pady=10, padx=10)  # center alignment

        Style().configure("My.TButton", font=('Helvetica', 12))
        # Style().map('My.TButton', background=[('active', 'red')])

        button1 = Button(self, text='AI Labeler', command=lambda: controller.show_frame(AI_Labeler, "AI_Labeler", branch), style='My.TButton')
        button1.pack(fill="both", pady=10, padx=200, expand=True)
        button2 = Button(self, text='Human Reviewer', command=lambda: controller.show_frame(Human_Reviewer, "Human_Reviewer", branch), style='My.TButton')
        button2.pack(fill="both", pady=10, padx=200, expand=True)
        button3 = Button(self, text='Human Reviser', command=lambda: controller.show_frame(Human_Reviser, "Human_Reviser", branch), style='My.TButton')
        button3.pack(fill="both", pady=10, padx=200, expand=True)
        button4 = Button(self, text='MainMenu', command=lambda: controller.show_frame(BranchMenu, "BranchMenu", 0), style='My.TButton')
        button4.pack(fill="both", pady=10, padx=200, expand=True)
        button5 = Button(self, text='Exit', command=lambda: controller.exit(), style='My.TButton')
        button5.pack(fill="both", pady=10, padx=200, expand=True)

class FaceMenu(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        # Style().configure("My.TLabel", background='#fff4f7')
        title = Label(self, text="Faical Landmarks Annotation", font="none 20 bold") #none 20 bold
        title.pack(pady=10, padx=10)  # center alignment

        Style().configure("My.TButton", font=('Helvetica', 12))
        # Style().map('My.TButton', background=[('active', 'red')])

        button1 = Button(self, text='AI Labeler', command=lambda: controller.show_frame(AI_Labeler, "AI_Labeler", branch), style='My.TButton')
        button1.pack(fill="both", pady=10, padx=200, expand=True)
        button2 = Button(self, text='Human Reviewer', command=lambda: controller.show_frame(Human_Reviewer, "Human_Reviewer", branch), style='My.TButton')
        button2.pack(fill="both", pady=10, padx=200, expand=True)
        button3 = Button(self, text='Human Reviser', command=lambda: controller.show_frame(Human_Reviser, "Human_Reviser", branch), style='My.TButton')
        button3.pack(fill="both", pady=10, padx=200, expand=True)
        button4 = Button(self, text='MainMenu', command=lambda: controller.show_frame(BranchMenu, "BranchMenu", 0), style='My.TButton')
        button4.pack(fill="both", pady=10, padx=200, expand=True)
        button5 = Button(self, text='Exit', command=lambda: controller.exit(), style='My.TButton')
        button5.pack(fill="both", pady=10, padx=200, expand=True)


class AI_Labeler(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text='AI Labeler', font=LARGE_FONT)
        label.grid(row=0, column=1, columnspan=1, sticky="e", padx=10, pady=10)
        # Choose and display resource folder
        self.output_res = Text(self, width=50, height=1, wrap="word", bg="white")
        self.output_res.grid(row=1, column=0, columnspan=4, sticky="nw",padx=10, pady=10)
        self.output_res.delete(0.0, END)
        btn_file = Button(self, text="Choose Resource", command=self.choose_resource)
        btn_file.grid(row=1, column=4, sticky="nw", padx=10, pady=10)

        txt_model = Label(self, text="Please choose a model:")
        txt_model.grid(row=2, column=0, sticky="nw", padx=10, pady=10)
        if branch == 1:
            self.Models = ["Hourglass", "Faster R-CNN"] # body pose estimators
            menu = BodyMenu
            name = "BodyMenu"
        elif branch == 2:
            self.Models = ["FAN"] # facial landmarks detector
            menu = FaceMenu
            name = "FaceMenu"
        else:
            self.Models = [] # empty
            menu = BranchMenu
            name = "BranchMenu"
        self.combo_model = Combobox(self, state="readonly", values=self.Models)
        self.combo_model.grid(row=2, column=1, sticky="nw", pady=10)
        self.combo_model.bind("<<ComboboxSelected>>", self.combo_callback)
        btn_AI = Button(self, text="Start Labeling", command=self.AI_labeling)
        btn_AI.grid(row=2, column=2, sticky="nw", padx=5, pady=10)

        # Keypoint annotation for different algorithms, as reference
        txt_explain = Label(self, text="KeyPoint annotation:")
        txt_explain.grid(row=3, column=0, columnspan=1, sticky="nw", padx=10, pady=10)
        self.ref = Text(self, width=60, height=4, wrap="word", bg="white")
        self.ref.grid(row=4, column=0, columnspan=60, sticky="nw",padx=10, pady=10)

        button1 = Button(self, text='Go Back', command=lambda: controller.show_frame(menu, name, branch))
        button1.grid(row=5, column=1, sticky="nw", padx=10, pady=10)

        button2 = Button(self, text='Exit', command=lambda: controller.exit())
        button2.grid(row=5, column=2, sticky="nw", padx=10, pady=10)

    def choose_resource(self):
        popup = Tk()
        popup.geometry("520x120")
        popup.wm_title("Choose Resource")
        label = Label(popup,
                      text=" If the resource are images, please choose directory. Otherwise, please choose a video file.")
        label.pack(side="top", pady=10)
        btn_dir = Button(popup, text="Choose directory", command=lambda: self.open_dir(popup))
        btn_video = Button(popup, text="Choose video", command=lambda: self.open_video(popup))
        btn_dir.pack()
        btn_video.pack()
        popup.mainloop()

    def combo_callback(self, event):
        # Show corresponding keypoints annotation
        print(self.combo_model.current(), self.combo_model.get())
        self.ref.delete(0.0, END)
        self.ref.insert(END, dict_model[event.widget.get()])

    def open_dir(self, popup):
        # load images folder
        popup.destroy()
        self.resource = filedialog.askdirectory()
        self.output_res.delete(0.0, END)
        self.output_res.insert(END, self.resource)

    def open_video(self, popup):
        # load video file
        popup.destroy()
        self.resource = filedialog.askopenfilename(initialdir='.',
                                                   filetypes=(("Video File", "*.mov"),("MP4", "*.mp4"),("AVI", "*.avi"),
                                                              ("All Files", "*.*")),
                                                   title="Choose a file")
        self.output_res.delete(0.0, END)
        self.output_res.insert(END, self.resource)

    def AI_labeling(self):
        if len(self.output_res.get("1.0", END)) == 1:
            messagebox.showwarning("Warning", "Please choose source!")

        # if resource is video, generate frame set for selected video
        if os.path.isfile(self.resource):
            print("Generating image set for video...")
            file_name = os.path.splitext(os.path.basename(self.resource))[0]
            folder = os.path.join(target, file_name)
            if not os.path.exists(folder):
                os.mkdir(folder)

            cap = cv2.VideoCapture(self.resource)
            currentframe = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    name = 'frame_' + str(currentframe) + '.jpg'
                    file = os.path.join(folder, name)
                    print('Creating...' + name)
                    cv2.imwrite(file, frame)
                    currentframe += 1
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()
            self.resource = folder

        # start with AI labeling
        if self.combo_model.get() == "Hourglass":
            model = "hg"
            AI_models.hourglass_model(self.resource, model)
            messagebox.showinfo("Info", "AI Labeling is done!")
        elif self.combo_model.get() == "Faster R-CNN":
            model = "fRCNN"
            AI_models.detectron2_model(self.resource, model)
            messagebox.showinfo("Info", "AI Labeling is done!")
        elif self.combo_model.get() == "FAN":                
            model = "fan"
            AI_models.FAN_model(self.resource, model)
            messagebox.showinfo("Info", "AI Labeling is done!")
        else:
            messagebox.showwarning("Warning", "Please choose proper model to label!")


class Human_Reviewer(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text='Human Reviewer', font=LARGE_FONT)
        label.grid(row=0, column=2, columnspan=5, sticky="nw", padx=10, pady=10)

        # Choose and display resource folder
        self.output_res = Text(self, width=55, height=1, wrap="word", bg="white")
        self.output_res.grid(row=1, column=0, columnspan=5, sticky="nw", padx=10, pady=10)
        btn_file = Button(self, text="Choose Images Folder",command=self.choose_resource)
        btn_file.grid(row=1, column=6, sticky="nw", padx=10, pady=10)

        # Choose and display AI keypoints file
        self.output_AI = Text(self, width=55, height=1, wrap="word", bg="white")
        self.output_AI.grid(row=2, column=0, columnspan=5, sticky="nw", padx=10, pady=10)
        btn_file = Button(self, text="Choose AI Result", command=self.choose_AIkpts)
        btn_file.grid(row=2, column=6, sticky="nw", padx=10, pady=10)

        btn_check = Button(self, text="Start Reviewing", command=self.review_label)
        btn_check.grid(row=3, column=1, sticky="nw", padx=5, pady=10)

        if branch == 1:
            menu = BodyMenu
            name = "BodyMenu"
        elif branch == 2:
            menu = FaceMenu
            name = "FaceMenu"
        else:
            menu = BranchMenu
            name = "BranchMenu"
        
        button1 = Button(self, text='Go Back',  # likewise StartPage
                         command=lambda: controller.show_frame(menu, name, branch))
        button1.grid(row=3, column=2, sticky="nw", padx=10, pady=10)

        button2 = Button(self, text='Exit',  # likewise StartPage
                         command=lambda: controller.exit())
        button2.grid(row=3, column=3, sticky="nw", padx=10, pady=10)

    def choose_resource(self):
        # load resource images/frames folder
        self.resource = filedialog.askdirectory()
        self.output_res.delete(0.0, END)
        self.output_res.insert(END, self.resource)

    def choose_AIkpts(self):
        # load AI keypoints file
        self.AIfile = filedialog.askopenfilename(initialdir='.',
                                                 filetypes=(("Pickle File", "*.pkl"), ("All Files", "*.*")),
                                                 title="Choose a file")
        self.output_AI.delete(0.0, END)
        self.output_AI.insert(END, self.AIfile)

        # extract model name
        filename, file_extension = os.path.splitext(os.path.basename(self.AIfile))
        string = filename.split('_')[-1]
        self.model = []
        print(string)
        if string == "hg":
            self.model = "Hourglass"
        elif string == "opencv":
            self.model = "OpenCV"
        elif string == "fRCNN":
            self.model = "Faster R-CNN"
        elif string == "fan":                   
            self.model = "FAN"

    def review_label(self):
        global dict_flags
        dict_flags.clear()

        # load images list
        types = ('*.jpg', '*.png', '*,jpeg')
        files_grabbed = []
        for files in types:
            files_grabbed.extend(glob.glob(os.path.join(self.resource, files)))
        self.im_list = sorted(files_grabbed)

        # load AI kpts array
        with open(self.AIfile, 'rb') as f:
            data = pickle.load(f)
        print(data)
        self.frames_kpts = data['all_keyps'][1]
        if self.model == "Hourglass":
            self.frames_boxes = data['all_boxes'][1]
        else:
            self.frames_boxes = data['all_boxes'][0]

        self.num_frames = len(self.frames_kpts)
        print("Total frames: ", self.num_frames)

        self.idx = 0
        self.show_figure()

    def show_figure(self):
        global flag, num_kpts, num_poses, txt_list, vis_idx, vis_pose_idx
        # initialize global variables
        flag = []
        txt_list = []
        num_kpts = 0
        num_poses = 0
        vis_idx = []
        vis_pose_idx = []

        # load current image
        im_name = os.path.basename(self.im_list[self.idx])
        img = cv2.imread(self.im_list[self.idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load current keypoints
        lists_kpts = self.frames_kpts[self.idx]

        if self.model == "Hourglass" or self.model == "Faster R-CNN" or self.model == "FAN":
            # single-person has only one pose
            vis_pose_idx = [0]
        # elif self.model == "Mask R-CNN":
        #   # multi-person has multiple poses
        #     # load current boxes
        #     lists_poses = self.frames_boxes[self.idx]
        #     vis_pose_idx = helpers.visposes(lists_poses)

        lists_vis, flatten_vis = helpers.viskpts(img, lists_kpts, vis_pose_idx, self.model)

        num_poses = len(lists_kpts)
        num_kpts = lists_kpts[0].shape[1]
        vis_idx = np.nonzero(flatten_vis)[0]

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 6]}, figsize=(10, 6))
        self.fig.canvas.set_window_title(im_name)

        # draw keypoints on image
        img = helpers.drawkpts(img, lists_kpts, lists_vis, self.model)

        # add text on image
        num_pose = len(lists_kpts)
        print("Total number of poses in current image: ", num_poses)
        if num_poses > 0:
            points = []
            for num in range(num_poses):
                for i in range(num_kpts):
                    if lists_vis[num][i] == 1:
                        x_kpts = lists_kpts[num][0, i]
                        y_kpts = lists_kpts[num][1, i]
                        points.append((int(x_kpts), int(y_kpts)))
                        plt.text(int(x_kpts), int(y_kpts),  str(num) + "_" + str(i), color='c', fontsize=12)
                    else:
                        points.append(None)

        # display keypoints reference in left subplot
        self.display_annotation(self.ax1)
        # show image with keypoints in right subplot
        self.ax2.imshow(img)
        self.ax2.set_axis_off()
        plt.tight_layout()
        # bind button and key with figure
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    def update_figure(self):
        global flag
        list_flag = []
        list = [1] * num_kpts * num_poses
        for i in range(len(vis_idx)):
            list[vis_idx[i]] = flag[i]
        for pose in range(num_poses):
            list_flag.append(list[pose * num_kpts:(pose + 1) * num_kpts])
        dict_flags[str(self.idx)] = list_flag
        print(dict_flags)

        self.idx = self.idx + 1
        if self.idx < len(self.frames_kpts):
            self.show_figure()
        else:
            messagebox.showinfo("Information", "All frames are reviewed and flags are saved!")
            helpers.savepkl(dict_flags, self.resource, "flag")
            plt.close()

    def on_click(self, event):
        global txt_list, flag, pt_list
        list = self.frames_kpts[self.idx]
        if len(flag) < len(vis_idx):
            if event.button == 1 and event.inaxes == self.ax2:
                txt = plt.text(event.xdata, event.ydata, str(vis_idx[len(flag)]%num_kpts)+'_R', color='b', fontsize=12)
                flag.append(1)
                txt_list.append(txt)
            elif event.button == 3 and event.inaxes == self.ax2:
                txt = plt.text(event.xdata, event.ydata, str(vis_idx[len(flag)]%num_kpts)+'_W', color='r', fontsize=12)
                point = plt.scatter(int(list[int(vis_idx[len(flag)]/num_kpts)][0, vis_idx[len(flag)]%num_kpts]),int(list[int(vis_idx[len(flag)]/num_kpts)][1, vis_idx[len(flag)]%num_kpts]), color='red', s = 50)
                flag.append(0)
                txt_list.append(txt)
                pt_list.append(point)

            self.fig.canvas.draw()
        else:
            print("Reviewing has done! Press 'y' to review next image, or press 'n' to review current image again.")

    def on_key(self, event):
        global txt_list, flag, vis_idx, pt_list
        if event.key == "d" and event.inaxes == self.ax2:
            if len(flag) < len(vis_idx):
                print("delete keypoint")
                txt = plt.text(event.xdata, event.ydata, str(vis_idx[len(flag)] % num_kpts) + '_D', color='r', fontsize=12)
                flag.append(-1)
                txt_list.append(txt)
                self.fig.canvas.draw()
            else:
                print("Reviewing has done! Press 'y' to review next image, or press 'n' to review current image again.")
        elif event.key == "i" and event.inaxes == self.ax2:
            print(len(vis_idx))
            print(num_kpts)
            if len(vis_idx) < num_kpts:
                if len(flag) == 0 and vis_idx[0] > 0:
                    print("insert keypoint")
                    vis_idx = np.insert(vis_idx, len(flag), 0)
                    txt = plt.text(event.xdata, event.ydata, str(vis_idx[len(flag)] % num_kpts) + '_I', color='r', fontsize=12)
                    point = plt.scatter(event.xdata, event.ydata, color='red', s=50)
                    flag.append(2)
                    txt_list.append(txt)
                    self.fig.canvas.draw()
                elif len(flag) > 0 and vis_idx[len(flag)-1]+1 < vis_idx[len(flag)]:
                    print("insert keypoint")
                    vis_idx = np.insert(vis_idx, len(flag), vis_idx[len(flag)-1]+1)
                    txt = plt.text(event.xdata, event.ydata, str(vis_idx[len(flag)] % num_kpts) + '_I', color='r', fontsize=12)
                    point = plt.scatter(event.xdata, event.ydata, color='red', s=50)
                    flag.append(2)
                    txt_list.append(txt)
                    self.fig.canvas.draw()
            else:
                if len(flag) < len(vis_idx):
                    print("Please continue to check!")
                else:
                    print("Reviewing has done! Press 'y' to review next image, or press 'n' to review current image again.")
        elif event.key == "u" and len(flag) != 0:
            print("check undo")
            txt_list[-1].remove()
            if flag[-1] == 2:
                vis_idx = np.delete(vis_idx, len(flag)-1)
                del txt_list[-1]
                del flag[-1]
            elif flag[-1] == 0:
                del txt_list[-1]
                pt_list[-1].remove()
                del pt_list[-1]
                del flag[-1]
            else:
                del txt_list[-1]
                del flag[-1]
            self.fig.canvas.draw()
        elif event.key == "y" and len(flag) == len(vis_idx):
            plt.close()
            print("Review next image")
            self.update_figure()
        elif event.key == "n":
            plt.close()
            print("Review current image again")
            self.show_figure()
            # answer = messagebox.askquestion("Review failed", "Are you sure label this image manually?")
            # if answer == 'yes':
            #     plt.close()
            #     flag = [0] * len(vis_idx)
            #     print("Review next image")
            #     self.updateFigure()
        else:
            print("Please continue to check!")

    def display_annotation(self, ax):
        str_list = dict_model[self.model].split(",")
        ax.set_axis_off()
        ax.set_ylim((0, len(str_list) + 2))
        for i in range(len(str_list)):
            ax.text(0, (len(str_list) - i), str_list[i], fontsize=9)
        ax.text(0, (len(str_list) + 1), "Keypoints Reference:", fontsize=12)


class Human_Reviser(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text='Human Reviser', font=LARGE_FONT)
        label.grid(row=0, column=2, columnspan=1, sticky="e", padx=10, pady=10)

        # Choose and display resource file
        self.output_res = Text(self, width=55, height=1, wrap="word", bg="white")
        self.output_res.grid(row=1, column=0, columnspan=5, sticky="nw", padx=10, pady=5)
        btn_file = Button(self, text="Choose Resource", command=self.choose_resource)
        btn_file.grid(row=1, column=6, sticky="nw", padx=2, pady=5)

        # Choose and display AI keypoints file
        self.output_AI = Text(self, width=55, height=1, wrap="word", bg="white")
        self.output_AI.grid(row=2, column=0, columnspan=5, sticky="nw", padx=10, pady=5)
        btn_file = Button(self, text="Choose AI Result", command=self.choose_AIkpts)
        btn_file.grid(row=2, column=6, sticky="nw", padx=2, pady=5)

        # Choose and display flags file
        self.output_Flags = Text(self, width=55, height=1, wrap="word", bg="white")
        self.output_Flags.grid(row=3, column=0, columnspan=5, sticky="nw", padx=10, pady=5)
        btn_file = Button(self, text="Choose Review Result", command=self.choose_flags)
        btn_file.grid(row=3, column=6, sticky="nw", padx=2, pady=5)

        btn_fix = Button(self, text="Start Revising", command=self.revise_label)
        btn_fix.grid(row=4, column=1, sticky="nw", padx=5, pady=10)

        if branch == 1:
            menu = BodyMenu
            name = "BodyMenu"
        elif branch == 2:
            menu = FaceMenu
            name = "FaceMenu"
        else:
            menu = BranchMenu
            name = "BranchMenu"

        button1 = Button(self, text='Go Back',  # likewise StartPage
                         command=lambda: controller.show_frame(menu, name, branch))
        button1.grid(row=4, column=2, sticky="nw", padx=40, pady=10)

        button2 = Button(self, text='Exit',  # likewise StartPage
                         command=lambda: controller.exit())
        button2.grid(row=4, column=3, sticky="nw", padx=10, pady=10)

    def choose_resource(self):
        # load resource images/frames folder
        self.resource = filedialog.askdirectory()
        self.output_res.delete(0.0, END)
        self.output_res.insert(END, self.resource)

    def choose_AIkpts(self):
        # load AI keypoints file
        self.AIfile = filedialog.askopenfilename(initialdir='.',
                                                 filetypes=(("Pickle File", "*.pkl"), ("All Files", "*.*")),
                                                 title="Choose a file")
        self.output_AI.delete(0.0, END)
        self.output_AI.insert(END, self.AIfile)

        # extract model name
        filename, file_extension = os.path.splitext(os.path.basename(self.AIfile))
        string = filename.split('_')[-1]
        self.model = []
        print(string)
        if string == "opencv":
            self.model = "OpenCV"
        elif string == "hg":
            self.model = "Hourglass"
        elif string == "fRCNN":
            self.model = "Faster R-CNN"
        elif string == "fan": 
            self.model = "FAN"

    def choose_flags(self):
        # load flags file
        self.Flagsfile = filedialog.askopenfilename(initialdir='.',
                                                 filetypes=(("Pickle File", "*.pkl"), ("All Files", "*.*")),
                                                 title="Choose a file")
        self.output_Flags.delete(0.0, END)
        self.output_Flags.insert(END, self.Flagsfile)

    def revise_label(self):
        global dict_flags,result
        dict_flags.clear()

        # load images list
        types = ('*.jpg', '*.png', '*.jpeg')
        files_grabbed = []
        for files in types:
            files_grabbed.extend(glob.glob(os.path.join(self.resource, files)))
        self.im_list = sorted(files_grabbed)

        # load AI kpts array
        with open(self.AIfile, 'rb') as f:
            data = pickle.load(f)


        if self.model == "Hourglass" or self.model == "Faster R-CNN" or self.model == "FAN":
            org_kpts = data['all_keyps'][1]
            if self.model == "Hourglass":
                frames_boxes = data['all_boxes'][1]
            else:
                frames_boxes = data['all_boxes'][0]
            result_kpts = copy.deepcopy(org_kpts)
            result['images'] = []
            result['all_keyps'] = [[], result_kpts]
            result['all_boxes'] = [[] for i in range(len(result_kpts))]
            self.frames_kpts = result['all_keyps'][1]
        # elif self.model == "Mask R-CNN":
        #     result = copy.deepcopy(data)
        #     self.frames_kpts = result['all_keyps'][1]
        #     self.frames_boxes = result['all_boxes'][1]

        self.num_frames = len(self.frames_kpts)
        print("Total frames: ", self.num_frames)

        # load reviewed flags array
        with open(self.Flagsfile, 'rb') as f:
            self.dict_flags = pickle.load(f)

        self.idx = 0
        self.show_flags()

    def show_flags(self):
        global num_kpts, num_poses, txt_list, plot_list, fix, fixed, vis_pose_idx, result, bbox
        # initialize global variables
        fix = []
        fixed = []
        txt_list = []
        plot_list = []
        num_kpts = 0
        num_poses = 0
        vis_pose_idx = []
        bbox = []

        # load current image
        im_name = os.path.basename(self.im_list[self.idx])
        img = cv2.imread(self.im_list[self.idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result['images'].append(im_name)

        # load current flags
        lists_flags = self.dict_flags[str(self.idx)]
        arr_flags = np.asarray(lists_flags)
        print("flag", arr_flags)

        num_poses = len(self.frames_kpts[self.idx])
        num_kpts = self.frames_kpts[self.idx][0].shape[1]

        # indexes of all false positive keypoints (incorrectly detected) for current image
        delete = np.where(arr_flags.flatten() == -1)[0]
        print('delete',delete)
        for i in range(len(delete)):
            pose = int(delete[i] / num_kpts)
            joint = delete[i] % num_kpts
            if self.model == "Hourglass" or self.model == "Faster R-CNN" or self.model == "FAN":
                # set the x,y to negative value
                result['all_keyps'][1][self.idx][pose][0][joint] = -45
                result['all_keyps'][1][self.idx][pose][1][joint] = -45
            # elif self.model == "Mask R-CNN":
            #     # decrease the confidence of false positive keypoints (less than 2)
            #     result['all_keyps'][1][self.idx][pose][2][joint] = 1.5

        # indexes of all false negative keypoints (incorrectly undetected) for current image
        insert = np.where(arr_flags.flatten() == 2)[0]
        print('insert', insert)
        for i in range(len(insert)):
            pose = int(insert[i] / num_kpts)
            joint = insert[i] % num_kpts
            if self.model == "Hourglass" or self.model == "Faster R-CNN" or self.model == "FAN":
                # set the x,y as coorindates of neck keypoint
                result['all_keyps'][1][self.idx][pose][0][joint] = result['all_keyps'][1][self.idx][pose][0][8]
                result['all_keyps'][1][self.idx][pose][1][joint] = result['all_keyps'][1][self.idx][pose][1][8]
            # elif self.model == "Mask R-CNN":
            #     # increase the confidence of false negative keypoints (more than 2)
            #     result['all_keyps'][1][self.idx][pose][2][joint] = 2.5

        # load current keypoints
        lists_kpts = self.frames_kpts[self.idx]
        print("kpt", lists_kpts)

        if self.model == "Hourglass" or self.model == "Faster R-CNN" or self.model == "FAN":
            # single-person has only one pose
            vis_pose_idx = [0]
        # elif self.model == "Mask R-CNN":
        #   # multi-person has multiple poses
        #     # load current boxes
        #     lists_poses = self.frames_boxes[self.idx]
        #     vis_pose_idx = helpers.visposes(lists_poses)
        lists_vis, flatten_vis = helpers.viskpts(img, lists_kpts, vis_pose_idx, self.model)
        # print("vis", lists_vis)
        flat_arr_flags = arr_flags.flatten()
        for i in range(flat_arr_flags.shape[0]):
            if flat_arr_flags[i] == 0 or flat_arr_flags[i] == 2:
                fix.append(i)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 6]}, figsize=(10, 6))
        self.fig.canvas.set_window_title(im_name)

        # draw keypoints on image
        img = helpers.drawkpts(img, lists_kpts, lists_vis, self.model)

        # add flags on image
        num_pose = len(lists_kpts)
        for num in range(num_pose):
            x_kpts = lists_kpts[num][0]
            y_kpts = lists_kpts[num][1]
            points = np.append([x_kpts], [y_kpts], axis=0)
            flags = lists_flags[num]
            for i in range(points.shape[1]):
                if flags[i] == 0 or flags[i] == 2:
                    plt.plot(int(points[0, i]), int(points[1, i]), 'ro', markersize=8)
                    plt.text(int(points[0, i]), int(points[1, i]), str(num) + "_" + str(i), color='r', fontsize=12)

        # display keypoints reference in left subplot
        self.display_annotation(self.ax1)
        # show image with keypoints and flags in right subplot
        self.ax2.imshow(img)
        self.ax2.set_axis_off()
        plt.tight_layout()

        # bind button and key with figure
        self.fig.canvas.mpl_connect('button_press_event', self.onclick_revise)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey_revise)
        self.rs = RectangleSelector(self.ax2, self.line_select_callback,
                                        drawtype='box', useblit=False,
                                        button=[1],  # don't use middle button
                                        minspanx=5, minspany=5,
                                        spancoords='pixels',
                                        interactive=True)
        self.rs.set_active(False)
        plt.show()

    def update_flags(self):
        global result
        for num in range(num_poses):
            result['all_keyps'][1][self.idx][num] = np.append(result['all_keyps'][1][self.idx][num], np.ones((1, num_kpts)), axis=0)
            result['all_boxes'][self.idx].append(bbox)
        for i in range(len(fixed)):
            pose = int(fix[i] / num_kpts)
            joint = fix[i] % num_kpts
            # print(result['all_keyps'][1][self.idx][pose][0][joint])
            result['all_keyps'][1][self.idx][pose][0][joint] = fixed[i][0]
            result['all_keyps'][1][self.idx][pose][1][joint] = fixed[i][1]
            result['all_keyps'][1][self.idx][pose][-1][joint] = fixed[i][2]

        if self.model == "Faster R-CNN":
            for num in range(num_poses):
                for joint in range(len(result['all_keyps'][1][self.idx][num][-1])):
                    if (result['all_keyps'][1][self.idx][num][-1][joint] == 1):
                        result['all_keyps'][1][self.idx][num][-1][joint] = 2
                    if (result['all_keyps'][1][self.idx][num][-1][joint] == 0):
                        result['all_keyps'][1][self.idx][num][-1][joint] = 1
                    if (result['all_keyps'][1][self.idx][num][0][joint] == -45 and result['all_keyps'][1][self.idx][num][1][joint] == -45):
                        result['all_keyps'][1][self.idx][num][-1][joint] = 0

        self.idx = self.idx + 1
        if self.idx < len(self.frames_kpts):
            self.show_flags()
        else:
            helpers.savepkl(result, self.resource, "gt")
            messagebox.showinfo("Information", "All frames are revised and keypoints are saved!")
            plt.close()

    def onclick_revise(self, event):
        global txt_list, fixed, plot_list
        if len(fixed) < len(fix):
            if event.button == 1 and event.inaxes == self.ax2:
                plot, = plt.plot(event.xdata, event.ydata, 'bo', markersize=8)
                txt = plt.text(event.xdata, event.ydata, str(fix[len(fixed)]%num_kpts)+'_vis',
                               horizontalalignment='right',
                               verticalalignment='bottom',
                               color='b', fontsize=12)
                fixed.append([event.xdata, event.ydata, 1])
                plot_list.append(plot)
                txt_list.append(txt)
            elif event.button == 3 and event.inaxes == self.ax2:
                plot, = plt.plot(event.xdata, event.ydata, 'bo', markersize=8)
                txt = plt.text(event.xdata, event.ydata, str(fix[len(fixed)]%num_kpts)+'_invis',
                               horizontalalignment='right',
                               verticalalignment='bottom',
                               color='b', fontsize=12)
                fixed.append([event.xdata, event.ydata, 0])
                plot_list.append(plot)
                txt_list.append(txt)
            self.fig.canvas.draw()
        else:
            print("Please select head bounding box.")
            self.rs.set_active(True)
            if bbox != []:
                print("Revising has done!, Please press 'y' to revise next image...")

    def onkey_revise(self, event):
        global txt_list, plot_list, fixed
        if event.key == "u" and len(fixed) != 0:
            print("revise undo")
            # print(plot_list[-1])
            plot_list[-1].remove()
            txt_list[-1].remove()
            self.fig.canvas.draw()
            del plot_list[-1]
            del txt_list[-1]
            del fixed[-1]
        elif event.key == "y" and fix == [] and bbox != []:
            plt.close()
            print("Revise next image")
            self.update_flags()
        elif event.key == "y" and len(fixed) == len(fix) and bbox != []:
            plt.close()
            print("Revise next image")
            self.update_flags()
        else:
            print("Please continue to revise!")

    def line_select_callback(self, eclick, erelease):
        global bbox
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        bbox = [x1, y1, x2, y2]
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))

    def display_annotation(self, ax):
        str_list = dict_model[self.model].split(",")
        ax.set_axis_off()
        ax.set_ylim((0, len(str_list) + 2))
        for i in range(len(str_list)):
            ax.text(0, (len(str_list) - i), str_list[i], fontsize=9)
        ax.text(0, (len(str_list) + 1), "Keypoints Reference:", fontsize=12)

if __name__ == '__main__':
    app = MainWindow()
    app.mainloop()
