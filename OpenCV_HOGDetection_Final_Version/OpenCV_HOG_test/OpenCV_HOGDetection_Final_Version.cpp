#include <iostream>
#include <cmath>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/ml/ml.hpp"
#include <opencv2/gpu/gpu.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>


#define BSrate 7  // num % 
#define coRate 0.5

using namespace cv;
using namespace std;

IplImage *img;
vector<IplImage *> people;
vector<IplImage *> people2;
vector<IplImage *> PeopleInFrame1, PeopleInFrame2;
vector<IplImage *> PeopleInFrame1_up, PeopleInFrame1_down;
vector<IplImage *> PeopleInFrame2_up, PeopleInFrame2_down;
vector<int> match_x1, match_y1, match_x2, match_y2;
int m_x1, m_y1, m_x2, m_y2;
int counter;

char * BackGroundName = "input_background.jpg";

// BS = BackGround Substruction

void BS(char * FrameName, int n) {
	IplImage* pFrame = NULL;
	IplImage* pFrImg = NULL;
	IplImage* pBkImg = NULL;

	CvMat* pFrameMat = NULL;
	CvMat* pFrMat = NULL;
	CvMat* pBkMat = NULL;

	pFrame = cvLoadImage(BackGroundName);
	pBkImg = cvCreateImage(cvSize(pFrame->width, pFrame->height), IPL_DEPTH_8U, 1);
	pFrImg = cvCreateImage(cvSize(pFrame->width, pFrame->height), IPL_DEPTH_8U, 1);

	pBkMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
	pFrMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
	pFrameMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);

	//轉化成單通道圖像再處理
	cvCvtColor(pFrame, pBkImg, CV_BGR2GRAY);
	cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY);

	cvConvert(pFrImg, pFrameMat);
	cvConvert(pFrImg, pFrMat);
	cvConvert(pFrImg, pBkMat);

	pFrame = cvLoadImage(FrameName);

	cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY);
	cvConvert(pFrImg, pFrameMat);
	//將當前的frame與背景相減 
	cvAbsDiff(pFrameMat, pBkMat, pFrMat);
	//二值化前景圖  
	cvThreshold(pFrMat, pFrImg, 60, 255.0, CV_THRESH_BINARY);
	//更新背景  
	cvRunningAvg(pFrameMat, pBkMat, 0.003, 0);
	//將背景轉為圖像格式，用以顯示  
	cvConvert(pBkMat, pBkImg);

	if (n == 1) {
		cvSaveImage("output_BS_F1.jpg", pFrImg);
		//進行Frame1的去背
	}
	else {
		cvSaveImage("output_BS_F2.jpg", pFrImg);
		//進行Frame2的去背
	}

	cvReleaseImage(&pFrImg);
	cvReleaseImage(&pBkImg);
	cvReleaseMat(&pFrameMat);
	cvReleaseMat(&pFrMat);
	cvReleaseMat(&pBkMat);
}



CvHistogram *createHist(IplImage *img, int n) {
	IplImage *temp = img;
	int dims = 1, sizes = 8;
	float range[] = { 0, 255 };
	float*ranges[] = { range };
	CvHistogram *Hist = cvCreateHist(dims, &sizes, CV_HIST_ARRAY, ranges, 1);
	cvClearHist(Hist);

	IplImage *imgBlue = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage *imgGreen = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage *imgRed = cvCreateImage(cvGetSize(img), 8, 1);
	cvSplit(img, imgBlue, imgGreen, imgRed, NULL); 
	
	if (n == 1) {
		cvCalcHist(&imgBlue, Hist, 0, 0);
		cvNormalizeHist(Hist, 1);
		return Hist;
	}
	else if (n == 2) {
		cvCalcHist(&imgGreen, Hist, 0, 0);
		cvNormalizeHist(Hist, 1);
		return Hist;
	}
	else if (n == 3) {
		cvCalcHist(&imgRed, Hist, 0, 0);
		cvNormalizeHist(Hist, 1);
		return Hist;
	}
}

void ShowSamePerson(string origin, string tar, int number, int textFlag){
	Mat src = imread(origin, CV_LOAD_IMAGE_COLOR);
	Mat roiImg = imread(tar, CV_LOAD_IMAGE_COLOR);
	Mat displayImg = src.clone();
	Mat result;
	result.create(src.rows - roiImg.rows + 1, src.cols - roiImg.cols + 1, CV_32FC1);

	matchTemplate(src, roiImg, result, CV_TM_SQDIFF_NORMED);
	double minVal;
	Point minLoc;
	minMaxLoc(result, &minVal, 0, &minLoc, 0);

	int colo = counter % 7;
	Scalar tmpS;
	switch (colo){
		case 0:
			tmpS = Scalar(0, 0,255);
			break;
		case 1:
			tmpS = Scalar(0,255,0);
			break;
		case 2:
			tmpS = Scalar(255,0,0);
			break;
		case 3:
			tmpS = Scalar(0,0,0);
			break;
		case 4:
			tmpS = Scalar(255, 255, 255);
			break;
		case 5:
			tmpS = Scalar(127,0,127);
			break;
		case 6:
			tmpS = Scalar(127, 127,0);
			break;
		default:
			tmpS = Scalar(0,0,0);
			break;
	}

	if (number == 1){
		rectangle(displayImg, Point(m_x1, m_y1), Point(m_x1 + roiImg.cols , m_y1 + roiImg.rows*2), tmpS, 3);
		
		//if (textFlag)
		{
			stringstream ss;
			ss << counter;
			putText(displayImg, ss.str(), Point(m_x1, m_y1), CV_FONT_HERSHEY_COMPLEX, 3, Scalar(0, 0, 0));
		}
		imwrite("Match1.jpg", displayImg);
	}
		
	else{
		rectangle(displayImg, Point(m_x2, m_y2), Point(m_x2 + roiImg.cols , m_y2 + roiImg.rows*2), tmpS, 3);
		
		//if (textFlag)
		{
			stringstream ss;
			ss << counter;
			putText(displayImg, ss.str(), Point(m_x2, m_y2), CV_FONT_HERSHEY_COMPLEX, 3, Scalar(0, 0, 0));
		}
		imwrite("Match2.jpg", displayImg);
	}
		
	
}

void Combine(){
	counter++;
	IplImage  *img1, *img2, *dst1, *dst2, *dst_big;
	CvRect rect1 = cvRect(0, 0, 640, 360);
	CvRect rect2 = cvRect(0, 360, 640, 360);
	img1 = cvLoadImage("Match1.jpg");
	img2 = cvLoadImage("Match2.jpg");
	dst1 = cvCreateImage(cvSize(640, 360), img1->depth, 3);
	dst2 = cvCreateImage(cvSize(640, 360), img2->depth, 3);
	dst_big = cvCreateImage(cvSize(640, 720), img2->depth, 3);
	cvResize(img1, dst1); 
	cvResize(img2, dst2);
	cvSetImageROI(dst_big, rect1); 
	cvCopy(dst1, dst_big);
	cvSetImageROI(dst_big, rect2);
	cvCopy(dst2, dst_big);
	cvResetImageROI(dst_big); 
	cvSaveImage("Combine.jpg", dst_big);
	cvNamedWindow("Find the same person");
	cvShowImage("Find the same person", dst_big);
	cvWaitKey();
	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	cvReleaseImage(&dst1);
	cvReleaseImage(&dst2);
	cvReleaseImage(&dst_big);
}

void compareImg(){
	for (int i1 = 0; i1 < PeopleInFrame1.size(); i1++) {
		
		cvSaveImage("temp1_up.jpg", PeopleInFrame1_up.at(i1));
		cvSaveImage("temp1_down.jpg", PeopleInFrame1_down.at(i1));
		IplImage *Image1_up = cvLoadImage("temp1_up.jpg", 1);
		IplImage *Image1_down = cvLoadImage("temp1_down.jpg", 1);
		CvHistogram *Histogram1_B_up = createHist(Image1_up, 1);
		CvHistogram *Histogram1_G_up = createHist(Image1_up, 2);
		CvHistogram *Histogram1_R_up = createHist(Image1_up, 3);
		CvHistogram *Histogram1_B_down = createHist(Image1_down, 1);
		CvHistogram *Histogram1_G_down = createHist(Image1_down, 2);
		CvHistogram *Histogram1_R_down = createHist(Image1_down, 3);

		for (int i2 = 0; i2 < PeopleInFrame2.size(); i2++) {
			cvSaveImage("temp2_up.jpg", PeopleInFrame2_up.at(i2));
			cvSaveImage("temp2_down.jpg", PeopleInFrame2_down.at(i2));
			IplImage *Image2_up = cvLoadImage("temp2_up.jpg", 1);
			IplImage *Image2_down = cvLoadImage("temp2_down.jpg", 1);
			CvHistogram *Histogram2_B_up = createHist(Image2_up, 1);
			CvHistogram *Histogram2_G_up = createHist(Image2_up, 2);
			CvHistogram *Histogram2_R_up = createHist(Image2_up, 3);
			CvHistogram *Histogram2_B_down = createHist(Image2_down, 1);
			CvHistogram *Histogram2_G_down = createHist(Image2_down, 2);
			CvHistogram *Histogram2_R_down = createHist(Image2_down, 3);

			double correal_up =
				(cvCompareHist(Histogram1_R_up, Histogram2_R_up, CV_COMP_CORREL) +
				cvCompareHist(Histogram1_G_up, Histogram2_G_up, CV_COMP_CORREL) +
				cvCompareHist(Histogram1_B_up, Histogram2_B_up, CV_COMP_CORREL)) / 3;
			cout << endl << "Correlation of Upper Body : " << correal_up * 100 << " %" << endl;
			double correal_down =
				(cvCompareHist(Histogram1_R_down, Histogram2_R_down, CV_COMP_CORREL) +
				cvCompareHist(Histogram1_G_down, Histogram2_G_down, CV_COMP_CORREL) +
				cvCompareHist(Histogram1_B_down, Histogram2_B_down, CV_COMP_CORREL)) / 3;
			cout << "Correlation of Lower Body : " << correal_down * 100 << " %" << endl;
			if (correal_up >= coRate && correal_down >= coRate){
				cout << "***Find the same person !***" << endl;
				m_x1 = match_x1.at(i1);
				m_y1 = match_y1.at(i1);
				m_x2 = match_x2.at(i2);
				m_y2 = match_y2.at(i2);
				ShowSamePerson("Match1.jpg", "temp1_up.jpg",1,0);
				//ShowSamePerson("Match1.jpg", "temp1_down.jpg", 1,1);
				ShowSamePerson("Match2.jpg", "temp2_up.jpg",2,0);
				//ShowSamePerson("Match2.jpg", "temp2_down.jpg", 2,1);
				Combine();
			}
			
			cvNamedWindow("Image1", 1);
			cvNamedWindow("Image2", 1);
			cvShowImage("Image1", PeopleInFrame1.at(i1));
			cvShowImage("Image2", PeopleInFrame2.at(i2));
			cvWaitKey(0);
			cvDestroyWindow("Image1");
			cvDestroyWindow("Image2");
			cvReleaseImage(&Image1_up);
			cvReleaseImage(&Image1_down);
			cvReleaseImage(&Image1_up);
			cvReleaseImage(&Image2_down);
		}
	}
}



int HOG_Detect(char * imageName, int n)
{
	/*--------------------------- Initialize ---------------------------*/
	IplImage *BSimg;
	vector<Rect> found, foundRect, BSfoundRect;

	// Do Background Subtraction before HOG detection
	BS(imageName, n);

	img = cvLoadImage(imageName);
	if (n==1)
		cvSaveImage("Match1.jpg", img);
	else
		cvSaveImage("Match2.jpg", img);

	if (!img) {
		cout << "Can't find input frame" << endl;
		return -1;
	}

	if (n == 1) 
		BSimg = cvLoadImage("output_BS_F1.jpg");
	else
		BSimg = cvLoadImage("output_BS_F2.jpg");

	IplImage * noFoundRect = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3); //開RGB三個channel
	IplImage * FoundRect = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
	IplImage * BSFoundRect = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);

	for (int i = 0; i < noFoundRect->height; i++) {
		for (int j = 0; j < noFoundRect->width; j++) {
			CvScalar color = cvGet2D(img, i, j);
			cvSet2D(noFoundRect, i, j, CV_RGB(color.val[2], color.val[1], color.val[0]));
			cvSet2D(FoundRect, i, j, CV_RGB(color.val[2], color.val[1], color.val[0]));
			cvSet2D(BSFoundRect, i, j, CV_RGB(color.val[2], color.val[1], color.val[0]));
		}
	}
	/*--------------------------- Initialize ---------------------------*/


	/*--------------------------- HOG detect ---------------------------*/
	HOGDescriptor defaultHog;
	defaultHog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	defaultHog.detectMultiScale(img, found);
	/*--------------------------- HOG detect ---------------------------*/


	/*--------------------------- 框出行人 ---------------------------*/
	for (int i = 0; i < found.size(); i++) {
		Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++) {
			if (j != i && (r & found[j]) == r)
				break;
		}
		if (j == found.size()) {
			foundRect.push_back(r);
		}
	}

	for (int i = 0; i < foundRect.size(); i++){
		Rect r2 = foundRect[i];
		int width = r2.br().x - r2.tl().x;
		int heigh = r2.br().y - r2.tl().y;
		CvPoint v1x, v1y, v2x, v2y;

		v1x = cvPoint(r2.tl().x + (0.3*width), r2.tl().y + 0.2 * heigh);
		v1y = cvPoint(r2.br().x - (0.3*width), 0.500*r2.tl().y + 0.500*r2.br().y);
		v2x = cvPoint(r2.tl().x + (0.3*width), 0.505*r2.tl().y + 0.505*r2.br().y);
		v2y = cvPoint(r2.br().x - (0.3*width), r2.br().y - 0.2 * heigh);
		cvRectangle(FoundRect, v1x, v1y, cvScalar(0, 255, 0), 3);
		cvRectangle(FoundRect, v2x, v2y, cvScalar(0, 255, 0), 3);

		// 比對去背後的影像，以把HOG偵測的非人物件刪除
		int px, py, OuterPoints = (v1y.x - v1x.x) * (v2y.y - v1x.y);
		float InterPoints = 0.0;
		CvScalar p;
		for (px = v1x.x; px < v1y.x; px++) {
			for (py = v1x.y; py < v2x.y; py++) {
				p = cvGet2D(BSimg, py, px);
				int r = p.val[2];  //取得 Red
				// r > 250 , 則視為前景中的白點
				if (r > 250)
					InterPoints++;
			}
		}
		float rate = 100 * InterPoints / OuterPoints;
		/******************************** using for debug ***********************************/
		cout << "rate = " << rate << " %" << " and Stand Rate =  " << BSrate << " %" << endl;
		/******************************** using for debug ***********************************/
		if (rate > BSrate){
			cvRectangle(BSFoundRect, v1x, v1y, cvScalar(0, 255, 0), 3);
			cvRectangle(BSFoundRect, v2x, v2y, cvScalar(0, 255, 0), 3);
			/******************************** using for debug ***********************************/
			cvNamedWindow("HOG Detection debug", 0);
			cvResizeWindow("HOG Detection debug", 960, 540);
			cvShowImage("HOG Detection debug", BSFoundRect);
			//cvWaitKey(0);
			cvDestroyWindow("HOG Detection debug");
			/******************************** using for debug ***********************************/
		}
		else
			continue;

		// 擷取人像部分的圖片(ROI)
		Rect body, body_up, body_down;
		body.x = v1x.x;
		body.y = v1x.y;
		body.height = v2y.y - v1x.y;
		body.width = v1y.x - v1x.x;
		cvSetImageROI(img, body);
		cvSaveImage("ROI_Person.jpg", img);
		IplImage *B = cvLoadImage("ROI_Person.jpg");

		body_up.x = v1x.x;
		body_up.y = v1x.y;
		body_up.height = v1y.y - v1x.y;
		body_up.width = v1y.x - v1x.x;
		cvSetImageROI(img, body_up);
		cvSaveImage("ROI_Person_up.jpg", img);
		IplImage *U = cvLoadImage("ROI_Person_up.jpg");

		body_down.x = v2x.x;
		body_down.y = v2x.y;
		body_down.height = v2y.y - v2x.y;
		body_down.width = v2y.x - v2x.x;
		cvSetImageROI(img, body_down);
		cvSaveImage("ROI_Person_down.jpg", img);
		IplImage *D = cvLoadImage("ROI_Person_down.jpg");

		if (n == 1) {
			match_x1.push_back(v1x.x);
			match_y1.push_back(v1x.y);
			PeopleInFrame1.push_back(B);
			PeopleInFrame1_up.push_back(U);
			PeopleInFrame1_down.push_back(D);
		}			
		else {
			match_x2.push_back(v1x.x);
			match_y2.push_back(v1x.y);
			PeopleInFrame2.push_back(B);
			PeopleInFrame2_up.push_back(U);
			PeopleInFrame2_down.push_back(D);
		}		
		/*--------------------------- 框出行人 ---------------------------*/
	}


	/*--------------------------- 顯示結果視窗 ---------------------------*/
	if (n == 1)
		cout << "Frame1 HOG Detection" << endl;
	else
		cout << "Frame2 HOG Detection" << endl;
	
	cvNamedWindow("HOG Detection", 0);
	cvResizeWindow("HOG Detection", 960, 540);
	cvShowImage("HOG Detection", FoundRect);
	cvWaitKey(0);
	cvDestroyWindow("HOG Detection");
	cvReleaseImage(&FoundRect);

	cvNamedWindow("Improved HOG Detection", 0);
	cvResizeWindow("Improved HOG Detection", 960, 540);
	cvShowImage("Improved HOG Detection", BSFoundRect);
	cvWaitKey(0);
	cvDestroyWindow("Improved HOG Detection");
	cvReleaseImage(&BSFoundRect);
	/*--------------------------- 顯示結果視窗 ---------------------------*/

	return 0;
}

int main(){
	counter = 0;
	HOG_Detect("input_frame1.jpg", 1);
	HOG_Detect("input_frame2.jpg", 2);
	compareImg();
	cout << "-------------------------------------------------" << endl;
	cout << "There are " << counter << " match cases." << endl;
	cout << "-------------------------------------------------" << endl;
	system("pause");
	return 0;
}
