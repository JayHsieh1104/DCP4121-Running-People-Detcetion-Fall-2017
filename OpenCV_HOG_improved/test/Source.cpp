#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/ml/ml.hpp"
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace std;

int main(void) {
	//Mat img;
	IplImage *img;
	vector<Rect> found, foundRect;

	char * imageName = "BWtest.bmp";

	img = cvLoadImage(imageName);

	//if(!img.data){
	if (!img) {
		printf("找不到圖片\n");
		return -1;
	}

	/*---------------------------  建立結果視窗  ----------------------------*/
	// 另外用兩個IplImage存框框結果
	IplImage * noFoundRect = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3); //記得要開三個通道才是RGB三個channel
	IplImage * FoundRect = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 3);

	// 對兩個測試結果先用原本讀圖的pixel蓋上去
	for (int i = 0; i < noFoundRect->height; i++) {
		for (int j = 0; j < noFoundRect->width; j++) {
			CvScalar color = cvGet2D(img, i, j);
			cvSet2D(noFoundRect, i, j, CV_RGB(color.val[2], color.val[1], color.val[0]));
			cvSet2D(FoundRect, i, j, CV_RGB(color.val[2], color.val[1], color.val[0]));
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
	// 框出行人(重複的就不畫框)→尋找有重複遍布的框框
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
		//cvRectangle(FoundRect, r2.tl(), r2.br(), cvScalar(0, 255, 0), 3);
	}
	cout << "3.框出行人     ---------- succeeded" << endl;
	/*--------------------------- 框出行人 ------------------------------*/



	/*------------------------ 存檔與秀出視窗 ---------------------------*/
	// 存成檔案 output.jpg
	cvSaveImage("output.jpg", FoundRect);

	cvNamedWindow("before", CV_WINDOW_AUTOSIZE);
	cvShowImage("before", img);

	cvNamedWindow("after", CV_WINDOW_AUTOSIZE);
	cvShowImage("after", FoundRect);

	cout << "4.存檔與秀出視窗 -------- succeeded" << endl;

	waitKey(0);

	cvDestroyWindow("before");
	cvReleaseImage(&img);

	cvDestroyWindow("after");
	cvReleaseImage(&FoundRect);
	/*------------------------ 存檔與秀出視窗 ---------------------------*/

	return 0;
}
