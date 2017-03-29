#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/ml/ml.hpp"
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace cv;
using namespace std;

char * BackGroundName = "input_background.jpg";
char * FrameName = "input_frame.jpg";
char * OutputImageName = "output_HOG.jpg";
char * BSOutputImageName = "output_BS&HOG.jpg";

void BS(void) {

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

	cvSaveImage("BS.jpg", pFrImg);

	cvReleaseImage(&pFrImg);
	cvReleaseImage(&pBkImg);
	cvReleaseMat(&pFrameMat);
	cvReleaseMat(&pFrMat);
	cvReleaseMat(&pBkMat);

	cout << "0.進行Frame的去背 ------- succeeded" << endl;
}

int main(void) {
	
	BS();
	
	IplImage *img, *BSimg;
	vector<Rect> found, foundRect, BSfoundRect;

	img = cvLoadImage(FrameName);
	BSimg = cvLoadImage("BS.jpg");

	if (!img) {
		printf("找不到圖片\n");
		return -1;
	}

	/*---------------------------  建立結果視窗  ----------------------------*/
	// 另外用兩個IplImage存框框結果
	IplImage * noFoundRect = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3); //記得要開三個通道才是RGB三個channel
	IplImage * FoundRect = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);
	IplImage * BSFoundRect = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);

	// 對兩個測試結果先用原本讀圖的pixel蓋上去
	for (int i = 0; i < noFoundRect->height; i++) {
		for (int j = 0; j < noFoundRect->width; j++) {
			CvScalar color = cvGet2D(img, i, j);
			cvSet2D(noFoundRect, i, j, CV_RGB(color.val[2], color.val[1], color.val[0]));
			cvSet2D(FoundRect, i, j, CV_RGB(color.val[2], color.val[1], color.val[0]));
			cvSet2D(BSFoundRect, i, j, CV_RGB(color.val[2], color.val[1], color.val[0]));
		}
	}
	cout << "1.建立結果視窗 ---------- succeeded" << endl;
	/*---------------------------  建立結果視窗  ----------------------------*/


	/*---------------------------  HOG檢測  ----------------------------*/
	HOGDescriptor defaultHog;
	defaultHog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	defaultHog.detectMultiScale(img, found);
	cout << "2.HOG detect   ---------- succeeded" << endl;
	/*---------------------------  HOG檢測  ----------------------------*/


	/*----------------  尋找有無重複框出的部分，將之去除  --------------*/
	// 遍尋found尋找沒有被嵌套的長方形
	for (int i = 0; i < found.size(); i++) {
		Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++) {
			//如果對到嵌套的就跳出循環
			if (j != i && (r & found[j]) == r)
				break;
		}
		if (j == found.size())  {
			foundRect.push_back(r);
		}
	}
	/*----------------  尋找有無重複框出的部分，將之去除  --------------*/


	/*--------------------------- 框出行人 ------------------------------*/
	// 框出行人(重複的就不畫框)
	for (int i = 0; i < foundRect.size(); i++) {
		Rect r2 = foundRect[i];
		int width = r2.br().x - r2.tl().x;
		int heigh = r2.br().y - r2.tl().y;
		CvPoint v1x, v1y, v2x, v2y;

		v1x = cvPoint(r2.tl().x + (0.25*width), r2.tl().y + 0.1 * heigh);
		v1y = cvPoint(r2.br().x - (0.25*width), 0.500*r2.tl().y + 0.500*r2.br().y);
		v2x = cvPoint(r2.tl().x + (0.25*width), 0.510*r2.tl().y + 0.510*r2.br().y);
		v2y = cvPoint(r2.br().x - (0.25*width), r2.br().y - 0.1 * heigh);
		cvRectangle(FoundRect, v1x, v1y, cvScalar(0, 255, 0), 3);
		cvRectangle(FoundRect, v2x, v2y, cvScalar(0, 255, 0), 3);

		int InterPoints = 0, OuterPoints = width * heigh;
		CvScalar p;
		for (int px = r2.tl().x; px < r2.br().x; px++) {
			for (int py = r2.tl().y; py < r2.br().y; py++) {
				p = cvGet2D(BSimg, py, px);
				int r = p.val[2];  //取得紅色
				int g = p.val[1];  //取得綠色
				int b = p.val[0];  //取得藍色
				if (r != 0 && g != 0 && b != 0) 
					InterPoints++ ;
			}
		}
		float rate = (float)InterPoints / OuterPoints;
		if (rate > 0.1){
			cvRectangle(BSFoundRect, v1x, v1y, cvScalar(0, 255, 0), 3);
			cvRectangle(BSFoundRect, v2x, v2y, cvScalar(0, 255, 0), 3);
		}
	}
	cout << "3.框出行人     ---------- succeeded" << endl;
	/*--------------------------- 框出行人 ------------------------------*/



	/*------------------------ 存檔與秀出視窗 ---------------------------*/
	cvSaveImage(OutputImageName, FoundRect);
	cvSaveImage(BSOutputImageName, BSFoundRect);

	cvNamedWindow("Original Frame", 0);
	cvResizeWindow("Original Frame", 800, 450);
	cvShowImage("Original Frame", img);

	cvNamedWindow("After Hog Detect", 0);
	cvResizeWindow("After Hog Detect", 800, 450);
	cvShowImage("After Hog Detect", FoundRect);

	cvNamedWindow("After BS & Hog Detect", 0);
	cvResizeWindow("After BS & Hog Detect", 800, 450);
	cvShowImage("After BS & Hog Detect", BSFoundRect);

	cout << "4.存檔與秀出視窗 -------- succeeded" << endl;

	waitKey(0);

	cvDestroyWindow("Original Frame");
	cvReleaseImage(&img);

	cvDestroyWindow("After Hog Detect");
	cvReleaseImage(&FoundRect);

	cvDestroyWindow("After BS & Hog Detect");
	cvReleaseImage(&BSFoundRect);
	/*------------------------ 存檔與秀出視窗 ---------------------------*/

	return 0;
}
