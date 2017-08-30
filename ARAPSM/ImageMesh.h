#include <iostream>
#include <glut.h>
#include "glm.h"
#include "triangle.h"

using namespace std;

typedef struct _meshTriagnle{
	GLuint vinidices[3];
} meshTriangle;

typedef struct _meshEdge{
	GLuint eindices[2];
} meshEdge;

typedef struct _meshTexture{
	string name;
	GLuint id;
	GLfloat width;
	GLfloat height;
} meshTexture;

class ImageMesh{
public:
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
	ImageMesh(triangulateio* meshData, string imageName);

private:
	void Unitize(ImageMesh* model);

};