#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <fstream>
#include "ARAPSM.h"

using namespace std;
using namespace cv;

/* Class ImageTriangulator */

ImageTriangulator::~ImageTriangulator(){
	free(imgMesh.pointlist);
	free(imgMesh.pointattributelist);
	free(imgMesh.trianglelist);
	free(imgMesh.triangleattributelist);
	free(imgMesh.edgelist);
	free(imgMesh.edgemarkerlist);
}

void ImageTriangulator::SetWindow(string winName){
	windowName = winName;
}

vector<Point> ImageTriangulator::boundaryPoints(){
	return boundaryPt;
}

void ImageTriangulator::CreateImageMesh(Mat& _image, string _fileName){
	image = &_image;
	fileName = _fileName;
	imgWidth = image->cols;
	imgHeight = image->rows;
	draw = image->clone();
	imshow(windowName, draw);
	while (!SET_BOUNDARY){
		int key = waitKey(0);
		switch (key){
		case 'r':
			// reset
			draw = image->clone();
			imshow(windowName, draw);
			boundaryPt.clear();
			break;
		case 13:
			SET_BOUNDARY = true;
			//destroyWindow(windowName);
			break;
		case 27:
			exit(EXIT_SUCCESS);
			break;
		}
	}
	cout << "Boundary Set" << endl;
	Triangulate();
}

void ImageTriangulator::WriteObj(){
	FILE *output;
	//ofstream output;
	output = fopen("data/imageMesh.obj","w");
	cout << imgMesh.numberofpoints << " " << imgMesh.numberoftriangles << endl;
	cout << "write vertex&texture coord." << endl;
	for (int i = 0; i < imgMesh.numberofpoints; i++){
		fprintf(output, "v %f %f %f\n", imgMesh.pointlist[i * 2 + 0] / image->cols, imgMesh.pointlist[i * 2 + 1] / image->cols, 0.0);
		fprintf(output, "vt %f %f\n", imgMesh.pointlist[i * 2 + 0] / image->cols, imgMesh.pointlist[i * 2 + 1] / image->rows);
		//cout << "v " << imgMesh.pointlist[i * 2 + 0] << " " << imgMesh.pointlist[i * 2 + 1] << " 0.0" << endl;
		//output << "vt " << imgMesh->pointlist[i * 2 + 0] << " " << imgMesh->pointlist[i * 2 + 1] << endl;
	}
	cout << "write triangle faces" << endl;
	for (int i = 0; i < imgMesh.numberoftriangles; i++){
		fprintf(output, "f %d/%d %d/%d %d/%d\n", imgMesh.trianglelist[i * 3 + 0], imgMesh.trianglelist[i * 3 + 0], imgMesh.trianglelist[i * 3 + 1], imgMesh.trianglelist[i * 3 + 1], imgMesh.trianglelist[i * 3 + 2], imgMesh.trianglelist[i * 3 + 2]);
	}
	fclose(output);
}

void ImageTriangulator::Triangulate(){
	struct triangulateio in, mid, out, vorout;
	// initialize
	in.numberofpoints = boundaryPt.size();
	in.numberofpointattributes = 1;
	in.pointlist = (REAL *)malloc(in.numberofpoints * 2 * sizeof(REAL));
	in.pointattributelist = (REAL *)malloc(in.numberofpoints*in.numberofpointattributes*sizeof(REAL));
	for (int i = 0; i < boundaryPt.size(); i++){
		in.pointlist[i * 2 + 0] = boundaryPt[i].x;
		in.pointlist[i * 2 + 1] = boundaryPt[i].y;
		in.pointattributelist[i] = boundaryPt[i].x + boundaryPt[i].y;
	}
	in.pointmarkerlist = (int *)malloc(in.numberofpoints*sizeof(int));
	for (int i = 0; i < in.numberofpoints; i++){
		in.pointmarkerlist[i] = 2;
	}
	in.numberofsegments = in.numberofpoints;
	in.segmentlist = (int *)malloc(in.numberofsegments*sizeof(int));
	in.segmentmarkerlist = (int *)malloc(in.numberofsegments*sizeof(int));
	for (int i = 1; i <= in.numberofsegments; i++){
		in.segmentlist[(i - 1) * 2 + 0] = i;
		if(i<in.numberofpoints) in.segmentlist[(i - 1) * 2 + 1] = i + 1;
		else in.segmentlist[(i - 1) * 2 + 1] = 1;
		in.segmentmarkerlist[i - 1] = 2;
	}
	in.numberofholes = 0;
	in.numberofregions = 1;
	in.regionlist = (REAL*)malloc(in.numberofregions * in.numberofpoints * sizeof(REAL));
	for (int i = 0; i < in.numberofpoints; i++){
		in.regionlist[i] = 10.0;
	}

	printf("Input point set:\n\n");
	report(&in, 1, 0, 0, 0, 0, 0);

	// prepare output struct
	mid.pointlist = (REAL *) NULL;
	mid.pointattributelist = (REAL *) NULL;
	mid.pointmarkerlist = (int *) NULL;
	mid.trianglelist = (int *) NULL;
	mid.triangleattributelist = (REAL *) NULL;
	mid.segmentlist = (int *)NULL;
	mid.segmentmarkerlist = (int *)NULL;
	mid.edgelist = (int *) NULL;
	mid.edgemarkerlist = (int *) NULL;

	vorout.pointlist = (REAL *) NULL;
	vorout.pointattributelist = (REAL *)NULL;
	vorout.edgelist = (int *) NULL;
	vorout.normlist = (REAL *)NULL;

	triangulate("pAq32.5a2000ev", &in, &mid, &vorout);
	printf("Initial triangulation:\n\n");
	printf("Initial Voronoi diagram:\n\n");

	out.pointlist = (REAL *) NULL;
	out.pointattributelist = (REAL*)NULL;
	out.trianglelist = (int *) NULL;
	out.edgelist = (int *) NULL;
	
	triangulate("peBP", &mid, &out, (struct triangulateio *) NULL);
	printf("Refined triangulation:\n\n");
	report(&out, 0, 1, 0, 1, 1, 0);
	
	imgMesh = out;
	ShowMesh(&imgMesh);

	free(in.pointlist);
	free(in.pointattributelist);
	free(in.pointmarkerlist);
	//free(in.segmentmarkerlist);

	free(mid.pointlist);
	free(mid.pointattributelist);
	free(mid.pointmarkerlist);
	free(mid.trianglelist);
	free(mid.triangleattributelist);
	free(mid.segmentlist);
	free(mid.segmentmarkerlist);
	free(mid.edgelist);
	free(mid.edgemarkerlist);

	free(vorout.pointlist);
	free(vorout.pointattributelist);
	free(vorout.edgelist);
	free(vorout.normlist);
}

void ImageTriangulator::ShowMesh(struct triangulateio* mesh){
	for (int i = 0; i < mesh->numberofedges; i++){
		int pt1 = mesh->edgelist[i * 2 + 0] - 1, pt2 = mesh->edgelist[i * 2 + 1] - 1;
		Point v1(mesh->pointlist[pt1 * 2 + 0], mesh->pointlist[pt1 * 2 + 1]);
		Point v2(mesh->pointlist[pt2 * 2 + 0], mesh->pointlist[pt2 * 2 + 1]);
		//cout << "vertex1: " << v1 << " vertex2: " << v2 << endl;
		line(draw, v1, v2, RED, 2);
	}
	imshow("Mesh", draw);
}

void ImageTriangulator::onMouse(int event, int x, int y, int flags){
	if (flags == CV_EVENT_FLAG_LBUTTON){ // drag
		Point pt(x, y);
		if (find(boundaryPt.begin(), boundaryPt.end(), pt) == boundaryPt.end()){
			int count;
			for (count = 0; count < boundaryPt.size(); count++){
				Point dis = pt - boundaryPt[count];
				if (dis.dot(dis) < 400) break;
			}
			if (count == boundaryPt.size()){
				boundaryPt.push_back(pt);
				circle(draw, Point(x, y), 1, Scalar(0, 0, 255), 2);
				imshow("Draw Boundary", draw);
			}
		}
	}
}

void ImageTriangulator::report(struct triangulateio *io, int markers, int reporttriangles, int reportneighbors, int reportsegments, int reportedges, int reportnorms){
	int i, j;

	for (i = 0; i < io->numberofpoints; i++) {
		printf("Point %4d:", i);
		for (j = 0; j < 2; j++) {
			printf("  %.6g", io->pointlist[i * 2 + j]);
		}
		if (io->numberofpointattributes > 0) {
			printf("   attributes");
		}
		for (j = 0; j < io->numberofpointattributes; j++) {
			printf("  %.6g",
				io->pointattributelist[i * io->numberofpointattributes + j]);
		}
		if (markers) {
			printf("   marker %d\n", io->pointmarkerlist[i]);
		}
		else {
			printf("\n");
		}
	}
	printf("\n");

	if (reporttriangles || reportneighbors) {
		for (i = 0; i < io->numberoftriangles; i++) {
			if (reporttriangles) {
				printf("Triangle %4d points:", i);
				for (j = 0; j < io->numberofcorners; j++) {
					printf("  %4d", io->trianglelist[i * io->numberofcorners + j]);
				}
				if (io->numberoftriangleattributes > 0) {
					printf("   attributes");
				}
				for (j = 0; j < io->numberoftriangleattributes; j++) {
					printf("  %.6g", io->triangleattributelist[i *
						io->numberoftriangleattributes + j]);
				}
				printf("\n");
			}
			if (reportneighbors) {
				printf("Triangle %4d neighbors:", i);
				for (j = 0; j < 3; j++) {
					printf("  %4d", io->neighborlist[i * 3 + j]);
				}
				printf("\n");
			}
		}
		printf("\n");
	}

	if (reportsegments) {
		for (i = 0; i < io->numberofsegments; i++) {
			printf("Segment %4d points:", i);
			for (j = 0; j < 2; j++) {
				printf("  %4d", io->segmentlist[i * 2 + j]);
			}
			if (markers) {
				printf("   marker %d\n", io->segmentmarkerlist[i]);
			}
			else {
				printf("\n");
			}
		}
		printf("\n");
	}

	if (reportedges) {
		for (i = 0; i < io->numberofedges; i++) {
			printf("Edge %4d points:", i);
			for (j = 0; j < 2; j++) {
				printf("  %4d", io->edgelist[i * 2 + j]);
			}
			if (reportnorms && (io->edgelist[i * 2 + 1] == -1)) {
				for (j = 0; j < 2; j++) {
					printf("  %.6g", io->normlist[i * 2 + j]);
				}
			}
			if (markers) {
				printf("   marker %d\n", io->edgemarkerlist[i]);
			}
			else {
				printf("\n");
			}
		}
		printf("\n");
	}
}

