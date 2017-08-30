#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <Windows.h>
#include <gl/GL.h>
#include "FreeImage.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "mtxlib.h"
#include "LeastSquaresSparseSolver.h"
#include "ARAPSM.h"

#define GL_BGRA 0x80E1

using namespace std;
using namespace cv;

/* Implement variable */
#define INPUTIMG "data/woody.jpg"

int WindWidth, WindHeight;
int last_x, last_y;
int selectedFeature = -1;
int selectedFIdx = -1;
int selectedTriangle = -1;
vector2 downPt, currentPt;
vector<int> featureList, featureTriangle; 
struct FeaturePt{
	vector2 pos;
	int triangleId;
	float w1, w2, w3;
};
vector<FeaturePt> featurePoint;
ImageMesh* imageMesh;

bool DeformAtUp = true;
bool Arbitrary = true;
bool RenderMesh = false;
bool RenderTexture = true;
bool FirstCompute = false;
bool SetRigidity = false;

#define W 100.0f
#define EDGEW 100.0f

vector<vector<int> > edgeNeighbor;
vector<float> weightEdge;
vector<vector2> edge;
vector<Eigen::MatrixXf> GtG, H;
LeastSquaresSparseSolver solver1, solver2;
float** b1 = nullptr;
float** b2 = nullptr;

/* Implement Function*/
float triangleArea(vector2 a, vector2 b, vector2 c);
void CalculatePointWeight(FeaturePt& pt);
int FindTriangle(ImageMesh* imgMesh, vector2 featurePos);
void vertexEdgeRelation(ImageMesh* imgMesh);
void setFirstCoefficient_k(ImageMesh* imgMesh, int k);
void setFirstCoefficient(ImageMesh* imgMesh);
void FirstStepMatrix(ImageMesh* imgMesh);
void FirstStepMesh();
void setSecondCoefficient_k(ImageMesh* imgMesh, int k, int ev1Idx, int ev2Idx);
void setSecondCoefficient(ImageMesh* imgMesh);
void SecondStepMatrix(ImageMesh* imgMesh);
void SecondStepMesh();

/* OpenGL Function */
void Reshape(int width, int height);
void Display(void);
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timf(int value);
vector2 Unproject(vector2 pos);

/* OpenCV Callback */
void onMouse(int event, int x, int y, int flags, void* param){
	ImageTriangulator *imgTriangle = (ImageTriangulator*)param;
	imgTriangle->onMouse(event, x, y, flags);
}

int main(int argc, char* argv[]){
	
	// A) Image Triangulation
	ImageTriangulator imgTriangle;
	string imageFile;
	imageFile = INPUTIMG;
	//printf("Input Image: ");
	//cin >> imageFile;
	Mat image = imread(imageFile);
	if (image.empty()){
		printf("Image not found!\n");
		return EXIT_FAILURE;
	}
	string windowName = "Draw Boundary";
	namedWindow(windowName,WINDOW_AUTOSIZE);
	setMouseCallback(windowName, onMouse, &imgTriangle);
	
	imgTriangle.SetWindow(windowName);
	imgTriangle.CreateImageMesh(image, imageFile);
	cout << "Mesh Generated." << endl;

	ImageMesh mesh(imgTriangle.getImageMesh(), imgTriangle.imageName(), imgTriangle.imgWidth, imgTriangle.imgHeight);
	imageMesh = &mesh;
	vertexEdgeRelation(imageMesh);
	//destroyAllWindows();

	// B) Shape Manipulation
	// OpenGL initialization
	FIBITMAP* pImage = FreeImage_Load(FreeImage_GetFileType(imageMesh->textureName.c_str(), 0), imageMesh->textureName.c_str());
	FIBITMAP *p32BitsImage = FreeImage_ConvertTo32Bits(pImage);
	imageMesh->texture[0].width = FreeImage_GetWidth(p32BitsImage);
	imageMesh->texture[0].height = FreeImage_GetHeight(p32BitsImage);

	WindWidth = WindHeight = 1000;
	glutInit(&argc, argv);
	glutInitWindowSize(WindWidth, WindHeight);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutCreateWindow("As Rigid As Possible Shape Manipulation");	
	
	glGenTextures(1, &imageMesh->texture[0].id);
	glBindTexture(GL_TEXTURE_2D, imageMesh->texture[0].id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageMesh->texture[0].width, imageMesh->texture[0].height, 0, GL_BGRA, GL_UNSIGNED_BYTE, (void*)FreeImage_GetBits(p32BitsImage));
	FreeImage_Unload(p32BitsImage);
	FreeImage_Unload(pImage);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

	glutReshapeFunc(Reshape);
	glutDisplayFunc(Display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glClearColor(0, 0, 0, 0);
	glutTimerFunc(40, timf, 0);
	
	glutMainLoop();

	return EXIT_SUCCESS;
}

/* Implement Function */
void vertexEdgeRelation(ImageMesh* imgMesh){
	// find neighbor vertex
	edgeNeighbor = vector<vector<int> >(imgMesh->numedges, vector<int>());
	for (int e = 0; e < imgMesh->numedges; e++){
		for (int i = 0; i < 2; i++){
			edgeNeighbor[e].push_back(imgMesh->edges[e].eindices[i]);
		}
	}

	vector<vector<int> >face(imgMesh->numtriangles, vector<int>());
	for (int f = 0; f < imgMesh->numtriangles; f++){
		for (int v = 0; v < 3; v++){
			int vIdx = imgMesh->triangles[f].vindices[v];
			face[f].push_back(vIdx);
		}
	}
	for (int f = 0; f < imgMesh->numtriangles; f++){
		for (int e = 0; e < imgMesh->numedges; e++){
			if (find(face[f].begin(), face[f].end(), imgMesh->edges[e].eindices[0]) != face[f].end()
				&& find(face[f].begin(), face[f].end(), imgMesh->edges[e].eindices[1]) != face[f].end()){
				for (int v = 0; v < 3; v++){
					if (imgMesh->triangles[f].vindices[v] != imgMesh->edges[e].eindices[0] && imgMesh->triangles[f].vindices[v] != imgMesh->edges[e].eindices[1]){
						edgeNeighbor[e].push_back(imgMesh->triangles[f].vindices[v]);
					}
				}
			}
		}
	}

	// Caculate GtGk & Hk first
	for (int k = 0; k < edgeNeighbor.size(); k++){
		// GtGk
		Eigen::MatrixXf Gk(edgeNeighbor[k].size() * 2, 2);
		for (int i = 0; i < edgeNeighbor[k].size(); i++){
			int evIdx = edgeNeighbor[k][i];
			vector2 ev(imgMesh->vertices[evIdx * 2 + 0], imgMesh->vertices[evIdx * 2 + 1]);
			Gk(i * 2, 0) = ev.x;
			Gk(i * 2, 1) = ev.y;
			Gk(i * 2 + 1, 0) = ev.y;
			Gk(i * 2 + 1, 1) = -ev.x;
		}

		int ev1Idx = edgeNeighbor[k][0], ev2Idx = edgeNeighbor[k][1];
		Eigen::MatrixXf E(2, 2);
		vector2 ek(imgMesh->vertices[ev1Idx * 2 + 0] - imgMesh->vertices[ev2Idx * 2 + 0], imgMesh->vertices[ev1Idx * 2 + 1] - imgMesh->vertices[ev2Idx * 2 + 1]);
		E(0, 0) = ek.x;
		E(0, 1) = ek.y;
		E(1, 0) = ek.y;
		E(1, 1) = -ek.x;
		edge.push_back(ek);
		weightEdge.push_back(1.0f);

		Eigen::MatrixXf GtGk = Gk.transpose()*Gk;
		GtGk = GtGk.inverse()*Gk.transpose();
		Eigen::MatrixXf Hk = E*GtGk;
		Hk = -1 * Hk;
		Hk(0, 0) += 1.0f;
		Hk(0, 2) -= 1.0f;
		Hk(1, 1) += 1.0f;
		Hk(1, 3) -= 1.0f;
		//cout << "Hk:\n" << Hk << endl;
		GtG.push_back(GtGk);
		H.push_back(Hk);

	}

}

bool InArea(vector2 lt, vector2 rd, vector2 pos){
	if (pos.x<lt.x || pos.x>rd.x) return false;
	if (pos.y<lt.y || pos.y>rd.y) return false;
	return true;
}

float triangleArea(vector2 a, vector2 b, vector2 c){
	vector2 edge1 = b - a;
	vector2 edge2 = c - a;
	float area = 0.5f*abs((edge1.x*edge2.y - edge2.x*edge1.y));
	return area;
}

int FindTriangle(ImageMesh* imgMesh, vector2 featurePos){
	int selectedTriangle = -1;
	for (int i = 0; i < imgMesh->numtriangles; i++){
		auto triangle = imgMesh->triangles[i];
		int v1Idx = triangle.vindices[0], v2Idx = triangle.vindices[1], v3Idx = triangle.vindices[2];
		vector2 v1(imgMesh->vertices[v1Idx * 2 + 0], imgMesh->vertices[v1Idx * 2 + 1]);
		vector2 v2(imgMesh->vertices[v2Idx * 2 + 0], imgMesh->vertices[v2Idx * 2 + 1]);
		vector2 v3(imgMesh->vertices[v3Idx * 2 + 0], imgMesh->vertices[v3Idx * 2 + 1]);
		float a1 = triangleArea(featurePos, v1, v2);
		float a2 = triangleArea(featurePos, v2, v3);
		float a3 = triangleArea(featurePos, v3, v1);
		float a = triangleArea(v1, v2, v3);
		if ((a1 + a2 + a3)-a<FLT_EPSILON){
			selectedTriangle = i;
			featureTriangle.push_back(i);
		}
	}
	cout << selectedTriangle << endl;
	return selectedTriangle;
}

void CalculatePointWeight(FeaturePt& pt){
	auto triangle = imageMesh->triangles[pt.triangleId];
	int v1Idx = triangle.vindices[0], v2Idx = triangle.vindices[1], v3Idx = triangle.vindices[2];
	vector2 v1(imageMesh->vertices[v1Idx * 2 + 0], imageMesh->vertices[v1Idx * 2 + 1]);
	vector2 v2(imageMesh->vertices[v2Idx * 2 + 0], imageMesh->vertices[v2Idx * 2 + 1]);
	vector2 v3(imageMesh->vertices[v3Idx * 2 + 0], imageMesh->vertices[v3Idx * 2 + 1]);
	float a1 = triangleArea(pt.pos, v1, v2);
	float a2 = triangleArea(pt.pos, v2, v3);
	float a3 = triangleArea(pt.pos, v3, v1);
	float a = triangleArea(v1, v2, v3);
	pt.w3 = a1 / a;
	pt.w1 = a2 / a;
	pt.w2 = a3 / a;
	//cout << pt.w1 << " " << pt.w2 << " " << pt.w3 << endl;
}

void setFirstCoefficient_k(ImageMesh* imgMesh, int k){
	for (int i = 0; i < edgeNeighbor[k].size(); i++){
		int evIdx = edgeNeighbor[k][i];
		solver1.AddSysElement(k * 2, evIdx * 2, H[k](0, i * 2));
		solver1.AddSysElement(k * 2, evIdx * 2 + 1, H[k](0, i * 2 + 1));;
		solver1.AddSysElement(k * 2 + 1, evIdx * 2, H[k](1, i * 2));
		solver1.AddSysElement(k * 2 + 1, evIdx * 2 + 1, H[k](1, i * 2 + 1));;
	}
}

void setFirstCoefficient(ImageMesh* imgMesh){
	//cout << "Set coefficient" << endl;
	// set coefficient by edge
	for (int k = 0; k < imgMesh->numedges; k++){
		int ev1Idx = imgMesh->edges[k].eindices[0], ev2Idx = imgMesh->edges[k].eindices[1];
		setFirstCoefficient_k(imgMesh, k);
	}
}

void FirstStepMatrix(ImageMesh* imgMesh){
	// solver initailize
	solver1.ResetSolver(0, 0, 0);
	int rowNum = (imgMesh->numedges + featureList.size()) * 2;
	if (Arbitrary) rowNum = (imgMesh->numedges + featurePoint.size()) * 2;
	int colNum = imgMesh->numvertices * 2;
	solver1.Create(rowNum, colNum, 1);
	if (b1 == nullptr)
		b1 = new float*[1];
	else
		delete[] b1[0];
	b1[0] = new float[rowNum];
	memset(b1[0], 0.0f, sizeof(float)*rowNum);

	// set element
	// preserve edge rigidity
	setFirstCoefficient(imgMesh);
	// positional constriant
	if (Arbitrary){
		for (int i = 0; i < featurePoint.size(); i++){
			FeaturePt pt = featurePoint[i];
			auto triangle = imageMesh->triangles[pt.triangleId];
			int v1Idx = triangle.vindices[0], v2Idx = triangle.vindices[1], v3Idx = triangle.vindices[2];
			solver1.AddSysElement((imgMesh->numedges + i) * 2, v1Idx * 2, W*pt.w1);
			solver1.AddSysElement((imgMesh->numedges + i) * 2 + 1, v1Idx * 2 + 1, W*pt.w1);
			solver1.AddSysElement((imgMesh->numedges + i) * 2, v2Idx * 2, W*pt.w2);
			solver1.AddSysElement((imgMesh->numedges + i) * 2 + 1, v2Idx * 2 + 1, W*pt.w2);
			solver1.AddSysElement((imgMesh->numedges + i) * 2, v3Idx * 2, W*pt.w3);
			solver1.AddSysElement((imgMesh->numedges + i) * 2 + 1, v3Idx * 2 + 1, W*pt.w3);

			b1[0][(imgMesh->numedges + i) * 2] = W*pt.pos.x;
			b1[0][(imgMesh->numedges + i) * 2 + 1] = W*pt.pos.y;
		}
	}
	else{
		for (int i = 0; i < featureList.size(); i++){
			int vIdx = featureList[i];
			solver1.AddSysElement((imgMesh->numedges + i) * 2, vIdx * 2, W*1.0f);
			solver1.AddSysElement((imgMesh->numedges + i) * 2 + 1, vIdx * 2 + 1, W*1.0f);
			b1[0][(imgMesh->numedges + i) * 2] = W*imgMesh->vertices[vIdx * 2 + 0];
			b1[0][(imgMesh->numedges + i) * 2 + 1] = W*imgMesh->vertices[vIdx * 2 + 1];
		}
	}
	solver1.SetRightHandSideMatrix(b1);
	solver1.CholoskyFactorization();
}

void FirstStepMesh(){
	if (Arbitrary){
		FeaturePt pt = featurePoint[selectedFIdx];
		b1[0][(imageMesh->numedges + selectedFIdx) * 2] = W*pt.pos.x;
		b1[0][(imageMesh->numedges + selectedFIdx) * 2 + 1] = W*pt.pos.y;
	}
	else{
		int vIdx = featureList[selectedFIdx];
		b1[0][(imageMesh->numedges + selectedFIdx) * 2] = W*imageMesh->vertices[vIdx * 2 + 0];
		b1[0][(imageMesh->numedges + selectedFIdx) * 2 + 1] = W*imageMesh->vertices[vIdx * 2 + 1];
	}
	
	solver1.SetRightHandSideMatrix(b1);
	solver1.CholoskySolve(0);
	for (int i = 0; i < imageMesh->numvertices; i++){
		imageMesh->vertices[i * 2 + 0] = solver1.GetSolution(0, i * 2);
		imageMesh->vertices[i * 2 + 1] = solver1.GetSolution(0, i * 2 + 1);
	}
	for (int i = 0; i < featurePoint.size(); i++){
		FeaturePt fp = featurePoint[i];
		auto triangle = imageMesh->triangles[fp.triangleId];
		int v1Idx = triangle.vindices[0], v2Idx = triangle.vindices[1], v3Idx = triangle.vindices[2];
		featurePoint[i].pos.x = fp.w1*imageMesh->vertices[v1Idx * 2 + 0] + fp.w2*imageMesh->vertices[v2Idx * 2 + 0] + fp.w3*imageMesh->vertices[v3Idx * 2 + 0];
		featurePoint[i].pos.y = fp.w1*imageMesh->vertices[v1Idx * 2 + 1] + fp.w2*imageMesh->vertices[v2Idx * 2 + 1] + fp.w3*imageMesh->vertices[v3Idx * 2 + 1];
	}
}

void setSecondCoefficient_k(ImageMesh* imgMesh, int k, int ev1Idx, int ev2Idx){
	// Compute Transform Matrix Tk
	Eigen::MatrixXf Vk(edgeNeighbor[k].size() * 2, 1);
	for (int i = 0; i < edgeNeighbor[k].size(); i++){
		int evIdx = edgeNeighbor[k][i];
		vector2 ev(imgMesh->vertices[evIdx * 2 + 0], imgMesh->vertices[evIdx * 2 + 1]);
		Vk(i * 2, 0) = ev.x;
		Vk(i * 2 + 1, 0) = ev.y;
	}

	Eigen::MatrixXf CkSk = GtG[k] * Vk;
	Eigen::MatrixXf Tk(2, 2);
	float scale = 1.0f/(CkSk(0, 0)*CkSk(0, 0) + CkSk(1, 0)*CkSk(1, 0));
	Tk(0, 0) = CkSk(0, 0);
	Tk(0, 1) = CkSk(1, 0);
	Tk(1, 0) = -CkSk(1, 0);
	Tk(1, 1) = CkSk(0, 0);
	Tk *= sqrtf(scale);

	Eigen::MatrixXf Ek(2, 1);
	Ek(0, 0) = edge[k].x;
	Ek(1, 0) = edge[k].y;
	Ek = Tk*Ek;

	float w = weightEdge[k];
	b2[0][k] = w*Ek(0, 0);
	b2[1][k] = w*Ek(1, 0);

}

void setSecondCoefficient(ImageMesh* imgMesh){
	//cout << "Set coefficient" << endl;
	// set coefficient by edge
	for (int k = 0; k < imgMesh->numedges; k++){
		int ev1Idx = imgMesh->edges[k].eindices[0], ev2Idx = imgMesh->edges[k].eindices[1];
		setSecondCoefficient_k(imgMesh, k, ev1Idx, ev2Idx);
	}
}

void SecondStepMatrix(ImageMesh* imgMesh){
	// solver initailize
	solver2.ResetSolver(0, 0, 0);
	int rowNum = imgMesh->numedges + featureList.size();
	if (Arbitrary) rowNum = imgMesh->numedges + featurePoint.size();
	int colNum = imgMesh->numvertices;
	solver2.Create(rowNum, colNum, 2);
	if (b2 == nullptr)
		b2 = new float*[2];
	else
		delete[] b2[0];
	b2[0] = new float[rowNum];
	b2[1] = new float[rowNum];
	memset(b2[0], 0.0f, sizeof(float)*rowNum);
	memset(b2[1], 0.0f, sizeof(float)*rowNum);

	// set element
	// preserve edge length
	for (int k = 0; k < edgeNeighbor.size(); k++){
		int ev1Idx = edgeNeighbor[k][0], ev2Idx = edgeNeighbor[k][1];
		float w = weightEdge[k];
		solver2.AddSysElement(k, ev1Idx, w*1.0f);
		solver2.AddSysElement(k, ev2Idx, w*-1.0f);
	}
	setSecondCoefficient(imgMesh);


	// positional constriant
	if (Arbitrary){
		for (int i = 0; i < featurePoint.size(); i++){
			FeaturePt pt = featurePoint[i];
			auto triangle = imageMesh->triangles[pt.triangleId];
			int v1Idx = triangle.vindices[0], v2Idx = triangle.vindices[1], v3Idx = triangle.vindices[2];
			solver2.AddSysElement(imgMesh->numedges + i, v1Idx, W*pt.w1);
			solver2.AddSysElement(imgMesh->numedges + i, v2Idx, W*pt.w2);
			solver2.AddSysElement(imgMesh->numedges + i, v3Idx, W*pt.w3);
			b2[0][imgMesh->numedges + i] = W*pt.pos.x;
			b2[1][imgMesh->numedges + i] = W*pt.pos.y;
		}
	}
	else{
		for (int i = 0; i < featureList.size(); i++){
			int vIdx = featureList[i];
			//cout << (imgMesh->numberofedges + i) * 2 << endl;
			solver2.AddSysElement(imgMesh->numedges + i, vIdx, W*1.0f);
			b2[0][imgMesh->numedges + i] = W*imgMesh->vertices[vIdx * 2 + 0];
			b2[1][imgMesh->numedges + i] = W*imgMesh->vertices[vIdx * 2 + 1];
		}
	}
	
	solver2.SetRightHandSideMatrix(b2);
	solver2.CholoskyFactorization();

}

void SecondStepMesh(){
	setSecondCoefficient(imageMesh);
	if (Arbitrary){
		FeaturePt pt = featurePoint[selectedFIdx];
		b2[0][imageMesh->numedges + selectedFIdx] = W*pt.pos.x;
		b2[1][imageMesh->numedges + selectedFIdx] = W*pt.pos.y;
	}
	else{
		int vIdx = featureList[selectedFIdx];
		b2[0][imageMesh->numedges + selectedFIdx] = W*imageMesh->vertices[vIdx * 2 + 0];
		b2[1][imageMesh->numedges + selectedFIdx] = W*imageMesh->vertices[vIdx * 2 + 1];
	}
	//cout << "b2:" << endl;
	//for (int i = 0; i < out.numberofedges + featureList.size(); i++){
	//	cout << b2[0][i] << endl;
	//}
	solver2.SetRightHandSideMatrix(b2);
	solver2.CholoskySolve(0);
	solver2.CholoskySolve(1);
	for (int i = 0; i < imageMesh->numvertices; i++){
		//cout << mesh->vertices[i * 3 + 0] << " " << mesh->vertices[i * 3 + 1] << endl;
		imageMesh->vertices[i * 2 + 0] = solver2.GetSolution(0, i);
		imageMesh->vertices[i * 2 + 1] = solver2.GetSolution(1, i);
		//cout << mesh->vertices[i * 3 + 0] << " " << mesh->vertices[i * 3 + 1] << endl << endl;
	}
	for (int i = 0; i < featurePoint.size(); i++){
		FeaturePt fp = featurePoint[i];
		auto triangle = imageMesh->triangles[fp.triangleId];
		int v1Idx = triangle.vindices[0], v2Idx = triangle.vindices[1], v3Idx = triangle.vindices[2];
		featurePoint[i].pos.x = fp.w1*imageMesh->vertices[v1Idx * 2 + 0] + fp.w2*imageMesh->vertices[v2Idx * 2 + 0] + fp.w3*imageMesh->vertices[v3Idx * 2 + 0];
		featurePoint[i].pos.y = fp.w1*imageMesh->vertices[v1Idx * 2 + 1] + fp.w2*imageMesh->vertices[v2Idx * 2 + 1] + fp.w3*imageMesh->vertices[v3Idx * 2 + 1];
	}
	//system("pause");
}

/* OpenGL Function */
void Reshape(int width, int height)
{
	if (height == 0) height = 1;
	int base = min(width, height);
	glViewport(0, 0, width, height);

	WindWidth = width;
	WindHeight = height;
}

void Display(void)
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glPushMatrix();

	// render 2D mesh model
	glColor3f(1.0f, 1.0f, 1.0f);
	if(RenderTexture) glEnable(GL_TEXTURE_2D);
	glBegin(GL_TRIANGLES);
	for (int i = 0; i < imageMesh->numtriangles; i++){
		auto triangle = imageMesh->triangles[i];
		//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		if (Arbitrary){
			if (find(featureTriangle.begin(), featureTriangle.end(), i) != featureTriangle.end()) glColor3f(1.0f, 1.0f, 0.0f);
			else glColor3f(1.0f, 1.0f, 1.0f);
		}
		glTexCoord2fv(&imageMesh->texcoords[2 * triangle.vindices[0]]);
		glVertex2fv(&imageMesh->vertices[2 * triangle.vindices[0]]);
		glTexCoord2fv(&imageMesh->texcoords[2 * triangle.vindices[1]]);
		glVertex2fv(&imageMesh->vertices[2 * triangle.vindices[1]]);
		glTexCoord2fv(&imageMesh->texcoords[2 * triangle.vindices[2]]);
		glVertex2fv(&imageMesh->vertices[2 * triangle.vindices[2]]);
	}
	glEnd();
	glDisable(GL_TEXTURE_2D);
	// render wire mesh
	if (RenderMesh){
		glColor3f(0.0f, 0.0f, 0.0f);
		glLineWidth(1.0f);
		for (int i = 0; i < imageMesh->numtriangles; i++){
			auto triangle = imageMesh->triangles[i];
			glLineWidth(2.0f);
			glBegin(GL_LINE_LOOP);
			glVertex2fv(&imageMesh->vertices[2 * triangle.vindices[0]]);
			glVertex2fv(&imageMesh->vertices[2 * triangle.vindices[1]]);
			glVertex2fv(&imageMesh->vertices[2 * triangle.vindices[2]]);
			glEnd();
		}
	}

	// render weighted edge
	glColor3f(1.0f, 0.0f, 0.0f);
	for (int i = 0; i < weightEdge.size(); i++){
		if (weightEdge[i] - 1.0f>FLT_EPSILON){
			int e1 = edgeNeighbor[i][0], e2 = edgeNeighbor[i][1];
			glLineWidth(2.0f);
			glBegin(GL_LINES);
			glVertex2fv(&imageMesh->vertices[2 * e1]);
			glVertex2fv(&imageMesh->vertices[2 * e2]);
			glEnd();
		}
	}
	if (SetRigidity){
		glColor3f(1.0f, 1.0f, 0.0f);
		vector2 leftTop, rightDown;
		leftTop.x = min(downPt.x, currentPt.x);
		leftTop.y = min(downPt.y, currentPt.y);
		rightDown.x = max(downPt.x, currentPt.x);
		rightDown.y = max(downPt.y, currentPt.y);
		for (int i = 0; i < imageMesh->numedges; i++){
			vector2 e1(imageMesh->vertices[2 * edgeNeighbor[i][0] + 0], imageMesh->vertices[2 * edgeNeighbor[i][0] + 1]);
			vector2 e2(imageMesh->vertices[2 * edgeNeighbor[i][1] + 0], imageMesh->vertices[2 * edgeNeighbor[i][1] + 1]);
			if (InArea(leftTop, rightDown, e1) && InArea(leftTop, rightDown, e2)){
				glLineWidth(5.0f);
				glBegin(GL_LINES);
				glVertex2f(e1.x, e1.y);
				glVertex2f(e2.x, e2.y);
				glEnd();
			}
		}
	}

	// render feature
	glPointSize(5.0);
	glColor3f(0.0f, 1.0f, 0.0f);
	glBegin(GL_POINTS);
	for (int i = 0; i < featureList.size(); i++){
		int fIdx = featureList[i];
		glVertex2fv(&imageMesh->vertices[2 * fIdx]);
	}
	glEnd();
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_POINTS);
	for (int i = 0; i < featurePoint.size(); i++){
		glVertex2f(featurePoint[i].pos.x, featurePoint[i].pos.y);
	}
	glEnd();
	
	glPopMatrix();
	glFlush();
	glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'a':
		Arbitrary = !Arbitrary;
		imageMesh->Reset();
		solver1.ResetSolver(0, 0, 0);
		solver2.ResetSolver(0, 0, 0);
		featureList.clear();
		featurePoint.clear();
		featureTriangle.clear();
		break;
	case 'd':
		DeformAtUp = !DeformAtUp;
		break;
	case 'r':
		imageMesh->Reset();
		solver1.ResetSolver(0, 0, 0);
		solver2.ResetSolver(0, 0, 0);
		featureList.clear();
		featurePoint.clear();
		featureTriangle.clear();
		for (int i = 0; i < weightEdge.size(); i++){
			weightEdge[i] = 1.0f;
		}
		break;
	case 'm':
		RenderMesh = !RenderMesh;
		break;
	case 't':
		RenderTexture = !RenderTexture;
		break;
	case 27:
		// release
		solver1.ResetSolver(0, 0, 0);
		solver2.ResetSolver(0, 0, 0);
		if (b1 != NULL){
			delete[] b1[0];
			delete[] b1;

		}
		if (b2 != NULL){
			delete[] b2[0];
			delete[] b2[1];
			delete[] b2;
		}
		exit(EXIT_SUCCESS);
		break;
	default:
		break;
	}
}

vector2 Unproject(float x, float y){
	vector2 pos(x, -y);
	float w, h, cx, cy, s;
	w = WindWidth;
	h = WindHeight;
	cx = w / 2.0;
	cy = h / 2.0;
	pos.x = (pos.x - cx)*2.0 / w;
	pos.y = (pos.y + cy)*2.0 / h;
	return pos;
}

void mouse(int button, int state, int x, int y)
{

	// add feature
	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN && !SetRigidity)
	{
		int minIdx = 0;
		float minDis = 9999999.0f;
		vector2 pos = Unproject(x, y);
		if (Arbitrary){
			FeaturePt fp;
			fp.triangleId = FindTriangle(imageMesh, pos);
			if (fp.triangleId != -1){
				fp.pos = pos;
				CalculatePointWeight(fp);
				featureTriangle.push_back(fp.triangleId);
				featurePoint.push_back(fp);
			}

		}
		else{
			for (int i = 0; i < imageMesh->numvertices; i++)
			{
				vector2 pt(imageMesh->vertices[2 * i + 0], imageMesh->vertices[2 * i + 1]);
				float dis = (pos - pt).length();

				if (minDis > dis)
				{
					minDis = dis;
					minIdx = i;
				}
			}

			featureList.push_back(minIdx);
			cout << minIdx << endl;
		}
		FirstStepMatrix(imageMesh);
		SecondStepMatrix(imageMesh);
	}

	// manipulate feature
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN && !SetRigidity)
	{
		int minIdx = -1, minfIdx = -1;
		float minDis = 9999999.0f;
		vector2 pos = Unproject(x, y);
		if (Arbitrary){
			for (int i = 0; i < featurePoint.size(); i++)
			{
				FeaturePt fp = featurePoint[i];
				vector2 pt(fp.pos.x, fp.pos.y);
				float dis = (pos - pt).length();

				if (minDis > dis)
				{
					minDis = dis;
					minfIdx = i;
				}
			}
			selectedFIdx = minfIdx;
			
		}
		else{
			for (int i = 0; i < featureList.size(); i++)
			{
				int idx = featureList[i];
				vector2 pt(imageMesh->vertices[2 * idx + 0], imageMesh->vertices[2 * idx + 1]);
				float dis = (pos - pt).length();

				if (minDis > dis)
				{
					minDis = dis;
					minIdx = featureList[i];
					minfIdx = i;
				}
			}
			selectedFeature = minIdx;
			selectedFIdx = minfIdx;
			cout << selectedFeature << " " << selectedFIdx << endl;
		}
		
		if (DeformAtUp){
			if (Arbitrary){
				featurePoint[selectedFIdx].pos.x = pos.x;
				featurePoint[selectedFIdx].pos.y = pos.y;
			}
			else{
				featurePoint[selectedFeature].pos.x = pos.x;
				featurePoint[selectedFeature].pos.y = pos.y;
			}
			FirstStepMesh();
			SecondStepMesh();
		}

		//system("pause");
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP && !SetRigidity){
		//cout << "select cancel!" << endl;
		if (selectedFIdx != -1){
			if (DeformAtUp){
				SecondStepMesh();
				FirstCompute = true;
			}
		}
		selectedFeature = -1;
		selectedFIdx = -1;
	}

	// set rigidity
	if (button == GLUT_LEFT_BUTTON&&state == GLUT_DOWN){
		if (FirstCompute){
			downPt = Unproject(x, y);
			SetRigidity = true;
		}
		cout << SetRigidity << endl;
	}

	if (button == GLUT_LEFT_BUTTON&&state == GLUT_UP){
		currentPt = Unproject(x, y);
		vector2 leftTop, rightDown;
		leftTop.x = min(downPt.x, currentPt.x);
		leftTop.y = min(downPt.y, currentPt.y);
		rightDown.x = max(downPt.x, currentPt.x);
		rightDown.y = max(downPt.y, currentPt.y);
		for (int i = 0; i < edgeNeighbor.size(); i++){
			vector2 e1(imageMesh->vertices[2 * edgeNeighbor[i][0] + 0], imageMesh->vertices[2 * edgeNeighbor[i][0] + 1]);
			vector2 e2(imageMesh->vertices[2 * edgeNeighbor[i][1] + 0], imageMesh->vertices[2 * edgeNeighbor[i][1] + 1]);
			if (InArea(leftTop, rightDown, e1) && InArea(leftTop, rightDown, e2)){
				cout << "Inside Edge!" << endl;
				weightEdge[i] = 1000.0f;
			}
		}
		SetRigidity = false;
		SecondStepMatrix(imageMesh);
	}

	last_x = x;
	last_y = y;
}

void motion(int x, int y)
{
	//tbMotion(x, y);

	if (selectedFIdx != -1)
	{
		vector2 pos = Unproject(x, y);
		vector2 last_pos = Unproject(last_x, last_y);
		vector2 move = pos - last_pos;
		if (Arbitrary){
			featurePoint[selectedFIdx].pos.x = pos.x;
			featurePoint[selectedFIdx].pos.y= pos.y;
			//featurePoint[selectedFIdx].pos.x += move.x;
			//featurePoint[selectedFIdx].pos.y += move.y;

		}
		else{
			imageMesh->vertices[2 * selectedFeature + 0] = pos.x;
			imageMesh->vertices[2 * selectedFeature + 1] = pos.y;
			//imageMesh->vertices[2 * selectedFeature + 0] += move.x;
			//imageMesh->vertices[2 * selectedFeature + 1] += move.y;

		}
		
		FirstStepMesh();
		if (!DeformAtUp) SecondStepMesh();
		FirstCompute = true;
	}

	if (SetRigidity){
		currentPt = Unproject(x, y);
	}

	last_x = x;
	last_y = y;
}

void timf(int value)
{
	glutPostRedisplay();
	glutTimerFunc(1, timf, 0);
}

