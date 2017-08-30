#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <glut.h>
#include "glm.h"
#include "triangle.h"

using namespace std;
using namespace cv;

const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);
const Scalar GREEN = Scalar(0, 255, 0);

typedef struct _meshTriagnle{
	GLuint vindices[3];
} meshTriangle;

typedef struct _meshEdge{
	GLuint eindices[2];
} meshEdge;

typedef struct _meshTexture{
	GLuint id;
	GLfloat width;
	GLfloat height;
} meshTexture;

class ImageMesh{
public:
	triangulateio* mesh;
	
	GLuint numvertices;
	GLfloat* vertices;

	GLuint numtexcoords;
	GLfloat* texcoords;

	GLuint numtriangles;
	meshTriangle* triangles;

	GLuint numedges;
	meshEdge* edges;

	// texture
	GLuint numtextures;
	meshTexture* texture;
	string textureName;

	ImageMesh(){}
	~ImageMesh();
	ImageMesh(triangulateio* meshData);
	ImageMesh(triangulateio* meshData, string filename, float w, float h);
	void Reset();

private:
	void Unitize(ImageMesh* model);

};

class ImageTriangulator{
public:
	/*member variable*/
	int imgWidth, imgHeight;
	/*member function*/
	ImageTriangulator(){}
	~ImageTriangulator();
	void SetWindow(string winName);
	void CreateImageMesh(Mat& _image,string _filename);
	void WriteObj();
	void onMouse(int event, int x, int y, int flag);
	
	vector<Point> boundaryPoints();
	struct triangulateio* getImageMesh(){ return &imgMesh; }
	string imageName(){ return fileName; }

private:
	string windowName, fileName;
	const Mat* image;
	Mat draw;
	vector<Point> boundaryPt;
	struct triangulateio imgMesh;
	bool SET_BOUNDARY = false;

	void Triangulate();
	void ShowMesh(struct triangulateio* mesh);
	void report(struct triangulateio *io, int markers, int reporttriangles, int reportneighbors, int reportsegments, int reportedges, int reportnorms);
};

class MeshRender{
public:
	MeshRender();
};