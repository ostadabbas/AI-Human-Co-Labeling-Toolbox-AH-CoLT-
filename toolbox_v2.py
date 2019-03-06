import Tkinter as tk
from ttk import *
import tkMessageBox
from PIL import Image, ImageTk
import os
import tkFileDialog as filedialog
import cPickle as pickle
import AI_Labeler
import numpy as np
import cv2
import helpers
import tkFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import copy
import glob

# plt.rcParams['toolbar'] = 'None'

LARGE_FONT = ("Verdana", 12)  # font's family is Verdana, font's size is 12

# dictionary of keypoints annotation
dict_model = {
        # "OpenCV(image)":"[OpenCV(image)]  0: Head 1: Neck 2: R_Shoulder 3: R_Elbow 4: R_Wrist 5: L_Shoulder"
        #          " 6: L_Elbow 7: L_Wrist 8: R_Hip 9: R_Knee 10: R_Ankle"
        #          " 11: L_Hip 12: L_Knee 13: L_Ankle 14: Chest",
        "Mask R-CNN":" 0: Nose, 1: L_Eye, 2: R_Eye, (3: L_Ear), (4: R_Ear), 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow,"
                     " 8: R_Elbow, 9: L_Wrist, 10: R_Wrist, 11: L_Hip, 12: R_Hip, 13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle"}

# global variables
num_kpts = 0
num_poses = 0
txt_list = []
dict_flags = {}
flag = []
fix = []
fixed = []
vis_idx = []
vis_pose_idx = []
frame_title = None
result = {}

class MainWindow(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("AI Human Co-labeling Toolbox")  # set the title of the main window
        self.geometry("580x340")  # set size of the main window to 580X340 pixels

        # this container contains all the pages
        container = Frame(self)
        container.pack(side= "top", fill = "both", expand = True)
        container.grid_rowconfigure(0, weight=1)  # make the cell in grid cover the entire window
        container.grid_columnconfigure(0, weight=1)  # make the cell in grid cover the entire window
        self.frames = {}  # these are pages we want to navigate to

        for F,geometry in zip((MainMenu, AI_Labeler, Human_Reviewer, Human_Reviser), ("580x300", "580x330", "610x200", "610x200")):  # for each page
            frame = F(container, self)  # create the page
            self.frames[F] = (frame, geometry)  # store into frames
            frame.grid(row=0, column=0, sticky="nsew")  # grid it to container

            self.show_frame(MainMenu)  # let the first page is StartPage

    def show_frame(self, name):
        frame, geometry = self.frames[name]
        self.geometry(geometry)
        frame.tkraise()

    def exit(self):
        self.destroy()

class MainMenu(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        # Title
        title = Label(self, text="AI Human Co-labeling Toolbox", font="none 28 bold") #none 20 bold
        title.pack(pady=10, padx=10)  # center alignment
        subtitle = Label(self, text="--Augmented Cognition Lab", font="none 22") # none 16
        subtitle.pack(pady=6, padx=6)  # center alignment

        # button of each step
        # helv12 = tkFont.Font(family='Helvetica', size=12, weight=tkFont.BOLD) # defined the font of button
        button1 = Button(self, text='AI Labeler', command=lambda: controller.show_frame(AI_Labeler))
        button1.pack(fill=tk.BOTH, pady=10, padx=200, expand=True)
        button2 = Button(self, text='Human Reviewer', command=lambda: controller.show_frame(Human_Reviewer))
        button2.pack(fill=tk.BOTH, pady=10, padx=200, expand=True)
        button3 = Button(self, text='Human Reviser', command=lambda: controller.show_frame(Human_Reviser))
        button3.pack(fill=tk.BOTH, pady=10, padx=200, expand=True)
        button4 = Button(self, text='Exit', command=lambda: controller.exit())
        button4.pack(fill=tk.BOTH, pady=10, padx=200, expand=True)

class AI_Labeler(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text='AI Labeler', font=LARGE_FONT)
        label.grid(row=0, column=1, columnspan=1, sticky="NW", padx=10, pady=10)

        # Choose and display resource folder
        self.output_res = tk.Text(self, width=50, height=1, wrap="word", bg="white")
        self.output_res.grid(row=1, column=0, columnspan=4, sticky="NW",padx=10, pady=10)
        btn_file = Button(self, text="Choose Resource", command=self.chooseResource)
        btn_file.grid(row=1, column=4, sticky="NW", padx=10, pady=10)

        txt_model = Label(self, text="Please choose a model:")
        txt_model.grid(row=2, column=0, sticky="NW", padx=10, pady=10)
        self.Models = ["Mask R-CNN"]
        self.combo_model = Combobox(self, state="readonly", values=self.Models)
        self.combo_model.grid(row=2, column=1, sticky="NW", pady=10)
        self.combo_model.bind("<<ComboboxSelected>>", self.comboCallback)
        btn_AI = Button(self, text="Start Labeling", command=self.AILabeling)
        btn_AI.grid(row=2, column=2, sticky="NW", padx=5, pady=10)

        # Keypoint annotation for different algorithms, as reference
        txt_explain = Label(self, text="Key Points:")
        txt_explain.grid(row=3, column=0, columnspan=1, sticky="NW", padx=10, pady=10)
        self.ref = tk.Text(self, width=60, height=4, wrap="word", bg="white")
        self.ref.grid(row=4, column=0, columnspan=60, sticky="NW",padx=10, pady=10)

        button1 = Button(self, text='Main Menu',  # likewise StartPage
                     command=lambda: controller.show_frame(MainMenu))
        button1.grid(row=5, column=1, sticky="NW", padx=10, pady=10)

        button2 = Button(self, text='Exit',  # likewise StartPage
                         command=lambda: controller.exit())
        button2.grid(row=5, column=2, sticky="NW", padx=10, pady=10)

    def comboCallback(self, event):
        # Show corresponding keypoints annotation
        print(self.combo_model.current(), self.combo_model.get())
        self.ref.delete(0.0, tk.END)
        self.ref.insert(tk.END, dict_model[event.widget.get()])

    def chooseResource(self):
        popup = tk.Tk()
        popup.geometry("520x120")
        popup.wm_title("Choose Resource")
        label = Label(popup, text=" If the resource are images, please choose directory. Otherwise, please choose a video file.")
        label.pack(side="top", pady=10)
        btn_dir = Button(popup, text="Choose directory", command=lambda: self.openDir(popup))
        btn_video = Button(popup, text="Choose video", command=lambda: self.openVideo(popup))
        btn_dir.pack()
        btn_video.pack()
        popup.mainloop()

    def openDir(self, popup):
        # load images folder
        popup.destroy()
        self.resource = filedialog.askdirectory()
        self.output_res.delete(0.0, tk.END)
        self.output_res.insert(tk.END, self.resource)

    def openVideo(self, popup):
        # load video file
        popup.destroy()
        self.resource = filedialog.askopenfilename(initialdir='.',
                            filetypes =(("Video File", "*.mov"),("MP4", "*.mp4"),("AVI", "*.avi"),("All Files", "*.*")),
                           title = "Choose a file")
        self.output_res.delete(0.0, tk.END)
        self.output_res.insert(tk.END, self.resource)

    def AILabeling(self):
        if self.combo_model.get()=="OpenCV" and os.path.isdir(self.resource):
            model = "opencv"
            [img_AI, keypoints] = AI_Labeler.OpenCV_Model(self.resource, model)
            # self.image_tk = ImageTk.PhotoImage(image = self.resizeImage(Image.fromarray(img_AI)))
            # self.canvas.create_image(50, 0, image=self.image_tk, anchor=NW)
            # np.save(os.path.join(self.dir, self.filename), keypoints)

            # save kpts as pkl file
            tkMessageBox.showinfo("Info", "AI Labeling is done!")
        elif self.combo_model.get()=="DensePose":
            model = "densepose"
            AI_Labeler.DensePose_Model(self.resource, model)
            print("None")
            tkMessageBox.showinfo("Info", "AI Labeling is done!")
        elif self.combo_model.get()=="Mask R-CNN" and self.resource.endswith((".mp4", ".avi")):
            model = "detections"
            AI_Labeler.DetectAndTrack_Model(self.resource, model)
            tkMessageBox.showinfo("Info", "AI Labeling is done!")
        # elif self.combo_model.get()=="Openpose":
        #     model = "openpose"
        #     print("None")
        #     tkMessageBox.showinfo("Info", "AI Labeling is done!")
        else:
            tkMessageBox.showwarning("Warning", "Please choose proper model to label!")

class Human_Reviewer(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text='Human Reviewer', font=LARGE_FONT)
        label.grid(row=0, column=2, columnspan=5, sticky="NW", padx=10, pady=10)

        # Choose and display resource folder
        self.output_res = tk.Text(self, width=55, height=1, wrap="word", bg="white")
        self.output_res.grid(row=1, column=0, columnspan=5, sticky="NW", padx=10, pady=10)
        btn_file = Button(self, text="Choose Images Folder",command=self.chooseResource)
        btn_file.grid(row=1, column=6, sticky="NW", padx=10, pady=10)

        # Choose and display AI keypoints file
        self.output_AI = tk.Text(self, width=55, height=1, wrap="word", bg="white")
        self.output_AI.grid(row=2, column=0, columnspan=5, sticky="NW", padx=10, pady=10)
        btn_file = Button(self, text="Choose AI Result", command=self.chooseAIkpts)
        btn_file.grid(row=2, column=6, sticky="NW", padx=10, pady=10)

        # btn_kpts = Button(self, text="Show Keypoints", command=self.annotateKpts)
        # btn_kpts.grid(row=3, column=1, sticky="NW", padx=5, pady=10)

        btn_check = Button(self, text="Start Reviewing", command=self.reviewLabel)
        btn_check.grid(row=3, column=1, sticky="NW", padx=5, pady=10)

        # # Keypoint annotation for different algorithms, as reference
        # txt_explain = Label(self, text="Key Points:")
        # txt_explain.grid(row=4, column=0, columnspan=1, sticky="NW", padx=10)
        # self.ref = Text(self, width=60, height=4, wrap="word", bg="white")
        # self.ref.grid(row=5, column=0, columnspan=60, sticky="NW", padx=10)

        button1 = Button(self, text='Main Menu',  # likewise StartPage
                         command=lambda: controller.show_frame(MainMenu))
        button1.grid(row=3, column=2, sticky="NW", padx=10, pady=10)

        button2 = Button(self, text='Exit',  # likewise StartPage
                         command=lambda: controller.exit())
        button2.grid(row=3, column=3, sticky="NW", padx=10, pady=10)

    def chooseResource(self):
        # load resource images/frames folder
        self.resource = filedialog.askdirectory()
        self.output_res.delete(0.0, tk.END)
        self.output_res.insert(tk.END, self.resource)

    def chooseAIkpts(self):
        # load AI keypoints file
        self.AIfile = filedialog.askopenfilename(initialdir='.',
                                                   filetypes=(("Pickle File", "*.pkl"), ("All Files", "*.*")),
                                                   title="Choose a file")
        self.output_AI.delete(0.0, tk.END)
        self.output_AI.insert(tk.END, self.AIfile)

        # extract model name
        filename, file_extension = os.path.splitext(os.path.basename(self.AIfile))
        string = filename.split('_')[-1]
        self.model = []
        print(string)
        if string == "opencv":
            self.model = "OpenCV"
        elif string == "densepose":
            self.model = "DensePose"
        elif string == "detections":
            self.model = "Mask R-CNN"
        # elif string == "openpose":
        #     self.model = "Openpose"

    def reviewLabel(self):
        global dict_flags
        dict_flags.clear()

        # load images list
        self.im_list = sorted(glob.glob(os.path.join(self.resource, "*.jpg")))

        # load AI kpts array
        with open(self.AIfile, 'rb') as f:
            data = pickle.load(f)
        self.frames_kpts = data['all_keyps'][1]
        self.num_frames = len(self.frames_kpts)
        print("Total frames: ", self.num_frames)

        self.frames_boxes = data['all_boxes'][1]

        self.idx = 0
        self.showFigure()

    def showFigure(self):
        global flag, num_kpts, num_poses, txt_list, vis_idx, vis_pose_idx
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

        # load current boxes
        lists_poses = self.frames_boxes[self.idx]
        vis_pose_idx = helpers.VisPoses(lists_poses)

        # load current keypoints
        lists_kpts = self.frames_kpts[self.idx]
        lists_vis, flatten_vis = helpers.VisKpts(lists_kpts, vis_pose_idx)

        num_poses = len(lists_kpts)
        num_kpts = lists_kpts[0].shape[1]
        vis_idx = np.nonzero(flatten_vis)[0]

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 6]}, figsize=(10, 6))
        self.fig.canvas.set_window_title(im_name)

        # draw keypoints on image
        img = helpers.DrawKpts(img, lists_kpts, lists_vis, self.model)

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
                        # points = np.append([x_kpts], [y_kpts], axis=0)
                        plt.text(int(x_kpts), int(y_kpts),  str(num) + "_" + str(i), color='c', fontsize=10)
                    else:
                        points.append(None)

        # display keypoints reference in left subplot
        self.displayAnnotation(self.ax1)
        # show image with keypoints in right subplot
        self.ax2.imshow(img)
        self.ax2.set_axis_off()
        plt.tight_layout()
        # bind button and key with figure
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    def updateFigure(self):
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
            self.showFigure()
        else:
            tkMessageBox.showinfo("Information", "All frames are reviewed and flags are saved!")
            self.saveFlags()
            plt.close()

    def on_click(self, event):
        global txt_list, flag
        print(event.button)
        print(event.xdata)
        # if len(flag) < num_kpts * num_poses:
        if len(flag) < len(vis_idx):
            if event.button == 1 and event.inaxes == self.ax2:
                txt = plt.text(event.xdata, event.ydata, str(vis_idx[len(flag)]%num_kpts)+'_R', color='b', fontsize=10)
                flag.append(1)
                txt_list.append(txt)
            elif event.button == 3 and event.inaxes == self.ax2:
                txt = plt.text(event.xdata, event.ydata, str(vis_idx[len(flag)]%num_kpts)+'_W', color='r', fontsize=10)
                flag.append(0)
                txt_list.append(txt)
            elif event.button == 2 and event.inaxes == self.ax2:
                txt = plt.text(event.xdata, event.ydata, str(vis_idx[len(flag)]%num_kpts)+'_D', color='k', fontsize=10)
                flag.append(-1)
                txt_list.append(txt)
            self.fig.canvas.draw()
        else:
            print("Reviewing has done!, Please press 'y' to review next image...")

    def on_key(self, event):
        global txt_list, flag
        if event.key == "u" and len(flag) != 0:
            print("check undo")
            # txt.remove()
            txt_list[-1].remove()
            self.fig.canvas.draw()
            del txt_list[-1]
            del flag[-1]
        elif event.key == "y" and len(flag) == len(vis_idx):
            plt.close()
            print("Review next image")
            self.updateFigure()
        elif event.key == "n":
            answer = tkMessageBox.askquestion("Review failed", "Are you sure label this image manually?")
            if answer == 'yes':
                plt.close()
                flag = [0] * len(vis_idx)
                print("Review next image")
                self.updateFigure()
        else:
            print("Please continue to check!")

    def displayAnnotation(self, ax):
        str_list = dict_model[self.model].split(",")
        # ax.set_title('Keypoints Reference')
        ax.set_axis_off()
        ax.set_ylim((0, len(str_list) + 2))
        for i in range(len(str_list)):
            ax.text(0, (len(str_list) - i), str_list[i], fontsize=9)
        ax.text(0, (len(str_list) + 1), "Keypoints Reference:", fontsize=12)

    def annotateKpts(self):
        self.ref.delete(0.0, tk.END)
        self.ref.insert(tk.END, dict_model[self.model])

    def saveFlags(self):
        path = self.resource + "_flag.pkl"
        with open(path, 'wb') as f:
            pickle.dump(dict_flags, f, protocol=pickle.HIGHEST_PROTOCOL)

class Human_Reviser(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        label = Label(self, text='Human Reviser', font=LARGE_FONT)
        label.grid(row=0, column=2, columnspan=1, sticky="NW", padx=10, pady=10)

        # Choose and display resource file
        self.output_res = tk.Text(self, width=55, height=1, wrap="word", bg="white")
        self.output_res.grid(row=1, column=0, columnspan=5, sticky="NW", padx=10, pady=5)
        btn_file = Button(self, text="Choose Resource", command=self.chooseResource)
        btn_file.grid(row=1, column=6, sticky="NW", padx=2, pady=5)

        # Choose and display AI keypoints file
        self.output_AI = tk.Text(self, width=55, height=1, wrap="word", bg="white")
        self.output_AI.grid(row=2, column=0, columnspan=5, sticky="NW", padx=10, pady=5)
        btn_file = Button(self, text="Choose AI Result", command=self.chooseAIkpts)
        btn_file.grid(row=2, column=6, sticky="NW", padx=2, pady=5)

        # Choose and display flags file
        self.output_Flags = tk.Text(self, width=55, height=1, wrap="word", bg="white")
        self.output_Flags.grid(row=3, column=0, columnspan=5, sticky="NW", padx=10, pady=5)
        btn_file = Button(self, text="Choose Review Result", command=self.chooseFlags)
        btn_file.grid(row=3, column=6, sticky="NW", padx=2, pady=5)

        # btn_kpts = Button(self, text="Show Keypoints", command=self.annotateKpts)
        # btn_kpts.grid(row=4, column=1, sticky="NW", padx=5, pady=10)

        btn_fix = Button(self, text="Start Revising", command=self.reviseLabel)
        btn_fix.grid(row=4, column=1, sticky="NW", padx=5, pady=10)

        button1 = Button(self, text='Main Menu',  # likewise StartPage
                         command=lambda: controller.show_frame(MainMenu))
        button1.grid(row=4, column=2, sticky="NW", padx=10, pady=10)

        button2 = Button(self, text='Exit',  # likewise StartPage
                         command=lambda: controller.exit())
        button2.grid(row=4, column=3, sticky="NW", padx=10, pady=10)

    def chooseResource(self):
        # load resource images/frames folder
        self.resource = filedialog.askdirectory()
        self.output_res.delete(0.0, tk.END)
        self.output_res.insert(tk.END, self.resource)

    def chooseAIkpts(self):
        # load AI keypoints file
        self.AIfile = filedialog.askopenfilename(initialdir='.',
                                                 filetypes=(("Pickle File", "*.pkl"), ("All Files", "*.*")),
                                                 title="Choose a file")
        self.output_AI.delete(0.0, tk.END)
        self.output_AI.insert(tk.END, self.AIfile)

        # extract model name
        filename, file_extension = os.path.splitext(os.path.basename(self.AIfile))
        string = filename.split('_')[-1]
        self.model = []
        print(string)
        if string == "opencv":
            self.model = "OpenCV"
        elif string == "densepose":
            self.model = "DensePose"
        elif string == "detections":
            self.model = "Mask R-CNN"
        # elif string == "openpose":
        #     self.model = "Openpose"

    def chooseFlags(self):
        # load flags file
        self.Flagsfile = filedialog.askopenfilename(initialdir='.',
                                                 filetypes=(("Pickle File", "*.pkl"), ("All Files", "*.*")),
                                                 title="Choose a file")
        self.output_Flags.delete(0.0, tk.END)
        self.output_Flags.insert(tk.END, self.Flagsfile)

    def reviseLabel(self):
        global dict_flags,result
        dict_flags.clear()

        # load images list
        self.im_list = sorted(glob.glob(os.path.join(self.resource, "*.jpg")))


        with open(self.AIfile, 'rb') as f:
            data = pickle.load(f)
        result = copy.deepcopy(data)

        # load AI kpts array
        self.frames_kpts = result['all_keyps'][1]
        self.num_frames = len(self.frames_kpts)
        print("Total frames: ", self.num_frames)

        self.frames_boxes = result['all_boxes'][1]

        # load reviewed flags array
        with open(self.Flagsfile, 'rb') as f:
            self.dict_flags = pickle.load(f)

        self.idx = 0
        self.showFlags()

    def showFlags(self):
        global num_kpts, num_poses, txt_list, plot_list, fix, fixed, vis_idx, vis_pose_idx, result
        fix = []
        fixed = []
        txt_list = []
        plot_list = []
        num_kpts = 0
        num_poses = 0
        vis_idx = []
        vis_pose_idx = []

        # load current image
        im_name = os.path.basename(self.im_list[self.idx])
        img = cv2.imread(self.im_list[self.idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        num_poses = len(self.frames_kpts[self.idx])
        num_kpts = self.frames_kpts[self.idx][0].shape[1]

        # load current boxes
        lists_poses = self.frames_boxes[self.idx]
        vis_pose_idx = helpers.VisPoses(lists_poses)

        # load current flags
        lists_flags = self.dict_flags[str(self.idx)]
        print("ddd", lists_flags)
        arr_flags = np.asarray(lists_flags)

        # indexes of all unreasonable keypoints for current image
        delete = np.where(arr_flags.flatten() == -1)[0]

        # decrease the confidence of unreasonable keypoints (less than 2)
        for i in range(len(delete)):
            pose = delete[i] / num_kpts
            joint = delete[i] % num_kpts
            # print(result['all_keyps'][1][self.idx][pose][0][joint])
            result['all_keyps'][1][self.idx][pose][2][joint] = 1.5

        # load current keypoints
        lists_kpts = self.frames_kpts[self.idx]

        # generate displayable keypoints list
        lists_vis, flatten_vis = helpers.VisKpts(lists_kpts, vis_pose_idx)
        vis_idx = np.nonzero(flatten_vis)[0]

        # indexes of all wrong keypoints for current image
        fix = np.where(arr_flags.flatten() == 0)[0]

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 6]}, figsize=(10, 6))
        self.fig.canvas.set_window_title(im_name)

        # draw keypoints on image
        img = helpers.DrawKpts(img, lists_kpts, lists_vis, self.model)

        # add flags on image
        num_pose = len(lists_kpts)
        for num in range(num_pose):
            x_kpts = lists_kpts[num][0]
            y_kpts = lists_kpts[num][1]
            points = np.append([x_kpts], [y_kpts], axis=0)
            flags = lists_flags[num]
            for i in range(points.shape[1]):
                if flags[i] == 0:
                    plt.text(int(points[0, i]), int(points[1, i]), str(num) + "_" + str(i), color='r', fontsize=10)

        # display keypoints reference in left subplot
        self.displayAnnotation(self.ax1)
        # show image with keypoints and flags in right subplot
        self.ax2.imshow(img)
        self.ax2.set_axis_off()
        plt.tight_layout()

        if fix == []:
            print("Needn't revise. Revise next image...")
            plt.close()
            self.updateFlags()
        else:
            # bind button and key with figure
            self.fig.canvas.mpl_connect('button_press_event', self.onclickRevise)
            self.fig.canvas.mpl_connect('key_press_event', self.onkeyRevise)
            plt.show()

    def updateFlags(self):
        global result
        for i in range(len(fixed)):
            pose = fix[i] / num_kpts
            joint = fix[i] % num_kpts
            # print(result['all_keyps'][1][self.idx][pose][0][joint])
            result['all_keyps'][1][self.idx][pose][0][joint] = fixed[i][0]
            result['all_keyps'][1][self.idx][pose][1][joint] = fixed[i][1]

        self.idx = self.idx + 10
        if self.idx < len(self.frames_kpts):
            self.showFlags()
        else:
            tkMessageBox.showinfo("Information", "All frames are revised and keypoints are saved!")
            self.saveKpts()
            plt.close()

    def onclickRevise(self, event):
        global txt_list, fixed, plot_list
        if len(fixed) < len(fix):
            if event.button == 1 and event.inaxes == self.ax2:
                plot, = plt.plot(event.xdata, event.ydata, 'go', markersize=2)
                txt = plt.text(event.xdata, event.ydata, str(fix[len(fixed)]%num_kpts)+'_New', color='g', fontsize=10)
                fixed.append([event.xdata, event.ydata])
                plot_list.append(plot)
                txt_list.append(txt)
            self.fig.canvas.draw()
        else:
            print("Revising has done!, Please press 'y' to revise next image...")

    def onkeyRevise(self, event):
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
        elif event.key == "y" and fix == []:
            plt.close()
            print("Revise next image")
            self.updateFlags()
        elif event.key == "y" and len(fixed) == len(fix):
            plt.close()
            print("Revise next image")
            self.updateFlags()
        else:
            print("Please continue to revise!")


    def displayAnnotation(self, ax):
        str_list = dict_model[self.model].split(",")
        ax.set_axis_off()
        ax.set_ylim((0, len(str_list) + 2))
        for i in range(len(str_list)):
            ax.text(0, (len(str_list) - i), str_list[i], fontsize=9)
        ax.text(0, (len(str_list) + 1), "Keypoints Reference:", fontsize=12)

    def annotateKpts(self):
        self.ref.delete(0.0, tk.END)
        self.ref.insert(tk.END, dict_model[self.model])

    def saveKpts(self):
        path = self.resource + "_gt.pkl"
        with open(path, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()