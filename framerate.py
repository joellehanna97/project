import cv2
import os
from pytube import YouTube
import pytube

global file_size
file_size = 0


# on_progress_callback takes 4 parameters.
def progress_Check(stream = None, chunk = None, file_handle = None, remaining = None):
	#Gets the percentage of the file that has been downloaded.
	percent = (100*(file_size-remaining))/file_size
	#print("{:00.0f}% downloaded".format(percent))

if __name__ == '__main__' :
	f = open("list_link.txt", "r")
	#f = open("train_partition.txt", "r")
	for url in f:
		try:
			yt = YouTube(url, on_progress_callback=progress_Check)
			title  = yt.title
			stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
			print(stream)
			file_size = stream.filesize
			stream.download()
			title.replace('/','')
			title.replace('?','')
			title.replace(',','')
			print(file_size)
			len_video = yt.length

			print(title)

			video = cv2.VideoCapture(title + '.mp4')

			fps = video.get(cv2.CAP_PROP_FPS)

			print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
			print(len_video)
			middle_frame = int(len_video)*int(fps)/2

			print(middle_frame)


			if not os.path.exists(title):
			    os.mkdir(title)
			#path = '/Users/joellehanna/Desktop/Master/Semestre 2/Projet/Basket-video' + '/' + title
			path = '/media/saeed-lts5/Data-Saeed/SuperResolution/sports1m-dataset/frames' + '/' + title
			success,image = video.read()
			count = 0
			frame_nb = 0
			keep_loop = True
			while success:
			    if count == int(middle_frame) + 150:
			        print('reached!')
			        keep_loop = False
			        break
			    #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
			    if count > middle_frame :
			        frame_nb += 1
			        cv2.imwrite(os.path.join(path , "frame_%05d.jpg" % frame_nb), image)

			    success,image = video.read()
			    #print('Read a new frame: ', success)
			    count += 1

			video.release();
			#mydir = '/Users/joellehanna/Desktop/Master/Semestre 2/Projet/Basket-video'
			mydir = '/media/saeed-lts5/Data-Saeed/SuperResolution/sports1m-dataset/frames'
			filelist = [ f for f in os.listdir(mydir) if f.endswith(".mp4") ]
			for f in filelist:
			    os.remove(os.path.join(mydir, f))
		except pytube.exceptions.VideoUnavailable:
			print('The following video is unavailable: {}'.format(url))
