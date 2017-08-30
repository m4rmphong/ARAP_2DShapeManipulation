#include <stdio.h>
#include "ARAPSM.h"

using namespace std;

ImageMesh:: ImageMesh(triangulateio* meshData, string filename, float w, float h){
	/* allocate a new model */
	mesh = meshData;
	numvertices = meshData->numberofpoints;
	vertices = (GLfloat*)malloc(sizeof(GLfloat)* 2 * numvertices);
	numtexcoords = meshData->numberofpoints;
	texcoords = (GLfloat*)malloc(sizeof(GLfloat)* 2 * numvertices);;
	numtriangles = meshData->numberoftriangles;
	triangles = (meshTriangle*)malloc(sizeof(meshTriangle)* 2 * numvertices);;
	numedges = meshData->numberofedges;
	edges = (meshEdge*)malloc(sizeof(meshEdge)* 2 * numedges);

	for (int i = 0; i < meshData->numberofpoints; i++){
		vertices[i * 2 + 0] = (float)meshData->pointlist[i * 2 + 0];
		vertices[i * 2 + 1] = -(float)meshData->pointlist[i * 2 + 1];
	}

	for (int i = 0; i < meshData->numberoftriangles; i++){
		for (int k = 0; k < 3; k++){
			triangles[i].vindices[k] = meshData->trianglelist[3 * i + k] - 1;
		}
	}

	for (int i = 0; i < meshData->numberofedges; i++){
		for (int k = 0; k < 2; k++){
			edges[i].eindices[k] = meshData->edgelist[2 * i + k] - 1;
		}
	}

	for (int i = 0; i < meshData->numberofpoints; i++){
		texcoords[i * 2 + 0] = (float)meshData->pointlist[i * 2 + 0] / w;
		texcoords[i * 2 + 1] = -(float)meshData->pointlist[i * 2 + 1] / h;
	}

	Unitize(this);

	numtextures = 1;
	texture = (meshTexture*)malloc(sizeof(meshTexture)*numtextures);
	textureName = filename;
	texture->width = w;
	texture->height = h;

}

ImageMesh:: ImageMesh(triangulateio* meshData){
	/* allocate a new model */
	mesh = meshData;
	numvertices = meshData->numberofpoints;
	vertices = (GLfloat*)malloc(sizeof(GLfloat)* 2 * numvertices);
	numtexcoords = meshData->numberofpoints;
	texcoords = NULL;// (GLfloat*)malloc(sizeof(GLfloat)* 2 * numvertice);;
	numtriangles = meshData->numberoftriangles;
	triangles = (meshTriangle*)malloc(sizeof(meshTriangle)* 2 * numvertices);;
	numedges = meshData->numberofedges;
	edges = (meshEdge*)malloc(sizeof(meshEdge)* 2 * numedges);

	for (int i = 0; i < meshData->numberofpoints; i++){
		vertices[i * 2 + 0] = meshData->pointlist[i * 2 + 0];
		vertices[i * 2 + 1] = meshData->pointlist[i * 2 + 1];
	}

	for (int i = 0; i < meshData->numberoftriangles; i++){
		for (int k = 0; k < 3; k++){
			triangles[i].vindices[k] = meshData->trianglelist[3 * i + k] - 1;
		}
	}

	for (int i = 0; i < meshData->numberofedges; i++){
		for (int k = 0; k < 2; k++){
			edges[i].eindices[k] = meshData->edgelist[2 * i + k] - 1;
		}
	}
	Unitize(this);

	numtextures = 0;
	texture = NULL;
}

void ImageMesh::Reset(){
	for (int i = 0; i < mesh->numberofpoints; i++){
		vertices[i * 2 + 0] = (float)mesh->pointlist[i * 2 + 0];
		vertices[i * 2 + 1] = -(float)mesh->pointlist[i * 2 + 1];
	}
	Unitize(this);
}

void ImageMesh::Unitize(ImageMesh* model){
	float minX, minY, maxX, maxY;
	maxX = minX = model->vertices[0];
	maxY = minY = model->vertices[1];

	for (int i = 0; i < model->numvertices; i++){
		maxX = maxX>model->vertices[i * 2 + 0] ? maxX : model->vertices[i * 2 + 0];
		minX = minX<model->vertices[i * 2 + 0] ? minX : model->vertices[i * 2 + 0];
		maxY = maxY>model->vertices[i * 2 + 1] ? maxY : model->vertices[i * 2 + 1];
		minY = minY<model->vertices[i * 2 + 1] ? minY : model->vertices[i * 2 + 1];
	}

	float w, h, centerX, centerY, scale;
	w = maxX + minX;
	h = minY + maxY;
	centerX = w / 2.0;
	centerY = h / 2.0;
	scale = w > h ? 1.6 / w : 1.6 / h;

	for (int i = 0; i < model->numvertices; i++){
		model->vertices[i * 2 + 0] -= centerX;
		model->vertices[i * 2 + 0] *= scale;
		model->vertices[i * 2 + 1] -= centerY;
		model->vertices[i * 2 + 1] *= scale;
	}
}

ImageMesh::~ImageMesh(){
	if (this->vertices) free(this->vertices);
	if (this->texture) free(this->texture);
	if (this->triangles) free(this->triangles);
	if (this->edges) free(this->edges);
}