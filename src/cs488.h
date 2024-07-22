// =======================================
// CS488/688 base code
// (written by Toshiya Hachisuka)
// =======================================
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX


// OpenGL
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>


// image loader and writer
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


// linear algebra 
#include "linalg.h"
using namespace linalg::aliases;


// animated GIF writer
#include "gif.h"


// misc
#include <iostream>
#include <vector>
#include <cfloat>
#include <ctime>
#include <thread>


// main window
static GLFWwindow* globalGLFWindow;


// window size and resolution
// (do not make it too large - will be slow!)
constexpr int globalWidth = 512;
constexpr int globalHeight = 384;


// degree and radian
constexpr float PI = 3.14159265358979f;
constexpr float DegToRad = PI / 180.0f;
constexpr float RadToDeg = 180.0f / PI;


// for ray tracing
constexpr float Epsilon = 5e-5f;
static int maxLevel = 10;
constexpr float etaAir = 1.0f; // 1.00029f;

// for SAH BVH
constexpr float Cb = 1.0f;
constexpr float C0 = 1.0f;


// amount the camera moves with a mouse and a keyboard
constexpr float ANGFACT = 0.2f;
constexpr float SCLFACT = 0.1f;


// fixed camera parameters
constexpr float globalAspectRatio = float(globalWidth / float(globalHeight));
constexpr float globalFOV = 45.0f; // vertical field of view
constexpr float globalDepthMin = Epsilon; // for rasterization
constexpr float globalDepthMax = 100.0f; // for rasterization
constexpr float globalFilmSize = 0.032f; //for ray tracing
const float globalDistanceToFilm = globalFilmSize / (2.0f * tan(globalFOV * DegToRad * 0.5f)); // for ray tracing


// particle system related
bool globalEnableParticles = false;
constexpr float deltaT = 0.002f;
constexpr float3 globalGravity = float3(0.0f, -9.8f, 0.0f);
constexpr int globalNumParticles = 100;
constexpr float G = 6.6743e-11f;
constexpr float globalTheta = 100.0f;
// #define A3_BONUS_1 // Comment this out to turn off Extra 1 for assignment 3


// dynamic camera parameters
float3 globalEye = float3(0.0f, 0.0f, 1.5f);
float3 globalLookat = float3(0.0f, 0.0f, 0.0f);
float3 globalUp = normalize(float3(0.0f, 1.0f, 0.0f));
float3 globalViewDir; // should always be normalize(globalLookat - globalEye)
float3 globalRight; // should always be normalize(cross(globalViewDir, globalUp));
bool globalShowRaytraceProgress = false; // for ray tracing


// mouse event
static bool mouseLeftPressed;
static double m_mouseX = 0.0;
static double m_mouseY = 0.0;


// path tracing
static int numThreads = std::thread::hardware_concurrency();


// rendering algorithm
enum enumRenderType {
	RENDER_RASTERIZE,
	RENDER_RAYTRACE,
	RENDER_IMAGE,
	RENDER_PATHTRACE,
};
enumRenderType globalRenderType = RENDER_IMAGE;
int globalFrameCount = 0;
static bool globalRecording = false;
static GifWriter globalGIFfile;
constexpr int globalGIFdelay = 1;


// OpenGL related data (do not modify it if it is working)
static GLuint GLFrameBufferTexture;
static GLuint FSDraw;
static const std::string FSDrawSource = R"(
    #version 120

    uniform sampler2D input_tex;
    uniform vec4 BufInfo;

    void main()
    {
        gl_FragColor = texture2D(input_tex, gl_FragCoord.st * BufInfo.zw);
    }
)";
static const char* PFSDrawSource = FSDrawSource.c_str();



// fast random number generator based pcg32_fast
#include <stdint.h>
namespace PCG32 {
	static uint64_t mcg_state = 0xcafef00dd15ea5e5u;	// must be odd
	static uint64_t const multiplier = 6364136223846793005u;
	uint32_t pcg32_fast(void) {
		uint64_t x = mcg_state;
		const unsigned count = (unsigned)(x >> 61);
		mcg_state = x * multiplier;
		x ^= x >> 22;
		return (uint32_t)(x >> (22 + count));
	}
	float rand() {
		return float(double(pcg32_fast()) / 4294967296.0);
	}
}



// image with a depth buffer
// (depth buffer is not always needed, but hey, we have a few GB of memory, so it won't be an issue...)
class Image {
public:
	std::vector<float3> pixels;
	std::vector<float> depths;
	int width = 0, height = 0;

	static float toneMapping(const float r) {
		// you may want to implement better tone mapping
		return std::max(std::min(1.0f, r), 0.0f);
	}

	static float gammaCorrection(const float r, const float gamma = 1.0f) {
		// assumes r is within 0 to 1
		// gamma is typically 2.2, but the default is 1.0 to make it linear
		return pow(r, 1.0f / gamma);
	}

	void resize(const int newWdith, const int newHeight) {
		this->pixels.resize(newWdith * newHeight);
		this->depths.resize(newWdith * newHeight);
		this->width = newWdith;
		this->height = newHeight;
	}

	void clear() {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				this->pixel(i, j) = float3(0.0f);
				this->depth(i, j) = FLT_MAX;
			}
		}
	}

	Image(int _width = 0, int _height = 0) {
		this->resize(_width, _height);
		this->clear();
	}

	bool valid(const int i, const int j) const {
		return (i >= 0) && (i < this->width) && (j >= 0) && (j < this->height);
	}

	float& depth(const int i, const int j) {
		return this->depths[i + j * width];
	}

	float3& pixel(const int i, const int j) {
		// optionally can check with "valid", but it will be slow
		return this->pixels[i + j * width];
	}

	void load(const char* fileName) {
		int comp, w, h;
		float* buf = stbi_loadf(fileName, &w, &h, &comp, 3);
		if (!buf) {
			std::cerr << "Unable to load: " << fileName << std::endl;
			return;
		}

		this->resize(w, h);
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				this->pixels[i + j * width] = float3(buf[k], buf[k + 1], buf[k + 2]);
				k += 3;
			}
		}
		delete[] buf;
		printf("Loaded \"%s\".\n", fileName);
	}
	void save(const char* fileName) {
		unsigned char* buf = new unsigned char[width * height * 3];
		int k = 0;
		for (int j = height - 1; j >= 0; j--) {
			for (int i = 0; i < width; i++) {
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).x)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).y)));
				buf[k++] = (unsigned char)(255.0f * gammaCorrection(toneMapping(pixel(i, j).z)));
			}
		}
		stbi_write_png(fileName, width, height, 3, buf, width * 3);
		delete[] buf;
		printf("Saved \"%s\".\n", fileName);
	}
};

// main image buffer to be displayed
Image FrameBuffer(globalWidth, globalHeight);

// you may want to use the following later for progressive ray tracing
Image AccumulationBuffer(globalWidth, globalHeight);
unsigned int sampleCount = 0;



// keyboard events (you do not need to modify it unless you want to)
void keyFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		switch (key) {
			case GLFW_KEY_R: {
				if (globalRenderType == RENDER_RAYTRACE) {
					printf("(Switched to rasterization)\n");
					glfwSetWindowTitle(window, "Rasterization mode");
					globalRenderType = RENDER_RASTERIZE;
				} else if (globalRenderType == RENDER_RASTERIZE) {
					printf("(Switched to ray tracing)\n");
					AccumulationBuffer.clear();
					sampleCount = 0;
					glfwSetWindowTitle(window, "Ray tracing mode");
					globalRenderType = RENDER_RAYTRACE;
				}
			break;}

			case GLFW_KEY_ESCAPE: {
				glfwSetWindowShouldClose(window, GL_TRUE);
			break;}

			case GLFW_KEY_I: {
				char fileName[1024];
				sprintf(fileName, "output%d.png", int(1000.0 * PCG32::rand()));
				FrameBuffer.save(fileName);
			break;}

			case GLFW_KEY_F: {
				if (!globalRecording) {
					char fileName[1024];
					sprintf(fileName, "output%d.gif", int(1000.0 * PCG32::rand()));
					printf("Saving \"%s\"...\n", fileName);
					GifBegin(&globalGIFfile, fileName, globalWidth, globalHeight, globalGIFdelay);
					globalRecording = true;
					printf("(Recording started)\n");
				} else {
					GifEnd(&globalGIFfile);
					globalRecording = false;
					printf("(Recording done)\n");
				}
			break;}

			case GLFW_KEY_W: {
				globalEye += SCLFACT * globalViewDir;
				globalLookat += SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_S: {
				globalEye -= SCLFACT * globalViewDir;
				globalLookat -= SCLFACT * globalViewDir;
			break;}

			case GLFW_KEY_Q: {
				globalEye += SCLFACT * globalUp;
				globalLookat += SCLFACT * globalUp;
			break;}

			case GLFW_KEY_Z: {
				globalEye -= SCLFACT * globalUp;
				globalLookat -= SCLFACT * globalUp;
			break;}

			case GLFW_KEY_A: {
				globalEye -= SCLFACT * globalRight;
				globalLookat -= SCLFACT * globalRight;
			break;}

			case GLFW_KEY_D: {
				globalEye += SCLFACT * globalRight;
				globalLookat += SCLFACT * globalRight;
			break;}

			default: break;
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void mouseButtonFunc(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouseLeftPressed = true;
		} else if (action == GLFW_RELEASE) {
			mouseLeftPressed = false;
			if (globalRenderType == RENDER_RAYTRACE) {
				AccumulationBuffer.clear();
				sampleCount = 0;
			}
		}
	}
}



// mouse button events (you do not need to modify it unless you want to)
void cursorPosFunc(GLFWwindow* window, double mouse_x, double mouse_y) {
	if (mouseLeftPressed) {
		const float xfact = -ANGFACT * float(mouse_y - m_mouseY);
		const float yfact = -ANGFACT * float(mouse_x - m_mouseX);
		float3 v = globalViewDir;

		// local function in C++...
		struct {
			float3 operator()(float theta, const float3& v, const float3& w) {
				const float c = cosf(theta);
				const float s = sinf(theta);

				const float3 v0 = dot(v, w) * w;
				const float3 v1 = v - v0;
				const float3 v2 = cross(w, v1);

				return v0 + c * v1 + s * v2;
			}
		} rotateVector;

		v = rotateVector(xfact * DegToRad, v, globalRight);
		v = rotateVector(yfact * DegToRad, v, globalUp);
		globalViewDir = v;
		globalLookat = globalEye + globalViewDir;
		globalRight = cross(globalViewDir, globalUp);

		m_mouseX = mouse_x;
		m_mouseY = mouse_y;

		if (globalRenderType == RENDER_RAYTRACE) {
			AccumulationBuffer.clear();
			sampleCount = 0;
		}
	} else {
		m_mouseX = mouse_x;
		m_mouseY = mouse_y;
	}
}




class PointLightSource {
public:
	float3 position, wattage;
};



class Ray {
public:
	float3 o, d;
	Ray() : o(), d(float3(0.0f, 0.0f, 1.0f)) {}
	Ray(const float3& o, const float3& d) : o(o), d(d) {}
};



// uber material
// "type" will tell the actual type
// ====== implement it in A2, if you want ======
enum enumMaterialType {
	MAT_LAMBERTIAN,
	MAT_METAL,
	MAT_GLASS
};
class Material {
public:
	std::string name;

	enumMaterialType type = MAT_LAMBERTIAN;
	float eta = 1.0f;
	float glossiness = 1.0f;

	float3 Ka = float3(0.0f);
	float3 Kd = float3(0.9f);
	float3 Ks = float3(0.0f);
	float Ns = 0.0;

	// support 8-bit texture
	bool isTextured = false;
	unsigned char* texture = nullptr;
	int textureWidth = 0;
	int textureHeight = 0;

	Material() {};
	virtual ~Material() {};

	void setReflectance(const float3& c) {
		if (type == MAT_LAMBERTIAN) {
			Kd = c;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
	}

	float3 fetchTexture(const float2& tex) const {
		// repeating
		int x = int(tex.x * textureWidth) % textureWidth;
		int y = int(tex.y * textureHeight) % textureHeight;
		if (x < 0) x += textureWidth;
		if (y < 0) y += textureHeight;

		int pix = (x + y * textureWidth) * 3;
		const unsigned char r = texture[pix + 0];
		const unsigned char g = texture[pix + 1];
		const unsigned char b = texture[pix + 2];
		return float3(r, g, b) / 255.0f;
	}

	// float3 BRDF(const float3& wi, const float3& wo, const float3& n) const {
	float3 BRDF() const {
		float3 brdfValue = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// BRDF
			brdfValue = Kd / PI;
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return brdfValue;
	};

	float PDF(const float3& wGiven, const float3& wSample) const {
		// probability density function for a given direction and a given sample
		// it has to be consistent with the sampler
		float pdfValue = 0.0f;
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}
		return pdfValue;
	}

	float3 sampler(const float3& wGiven, float& pdfValue) const {
		// sample a vector and record its probability density as pdfValue
		float3 smp = float3(0.0f);
		if (type == MAT_LAMBERTIAN) {
			// empty
		} else if (type == MAT_METAL) {
			// empty
		} else if (type == MAT_GLASS) {
			// empty
		}

		pdfValue = PDF(wGiven, smp);
		return smp;
	}
};





class HitInfo {
public:
	float t; // distance
	float3 P; // location
	float3 N; // shading normal vector
	float2 T; // texture coordinate
	const Material* material; // const pointer to the material of the intersected object
};



// axis-aligned bounding box
class AABB {
private:
	float3 minp, maxp, size;

public:
	float3 get_minp() { return minp; };
	float3 get_maxp() { return maxp; };
	float3 get_size() { return size; };


	AABB() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	void reset() {
		minp = float3(FLT_MAX);
		maxp = float3(-FLT_MAX);
		size = float3(0.0f);
	}

	int getLargestAxis() const {
		if ((size.x > size.y) && (size.x > size.z)) {
			return 0;
		} else if (size.y > size.z) {
			return 1;
		} else {
			return 2;
		}
	}

	void fit(const float3& v) {
		if (minp.x > v.x) minp.x = v.x;
		if (minp.y > v.y) minp.y = v.y;
		if (minp.z > v.z) minp.z = v.z;

		if (maxp.x < v.x) maxp.x = v.x;
		if (maxp.y < v.y) maxp.y = v.y;
		if (maxp.z < v.z) maxp.z = v.z;

		size = maxp - minp;
	}

	float area() const {
		return (2.0f * (size.x * size.y + size.y * size.z + size.z * size.x));
	}


	bool intersect(HitInfo& minHit, const Ray& ray) const {
		// set minHit.t as the distance to the intersection point
		// return true/false if the ray hits or not
		float tx1 = (minp.x - ray.o.x) / ray.d.x;
		float ty1 = (minp.y - ray.o.y) / ray.d.y;
		float tz1 = (minp.z - ray.o.z) / ray.d.z;

		float tx2 = (maxp.x - ray.o.x) / ray.d.x;
		float ty2 = (maxp.y - ray.o.y) / ray.d.y;
		float tz2 = (maxp.z - ray.o.z) / ray.d.z;

		if (tx1 > tx2) {
			const float temp = tx1;
			tx1 = tx2;
			tx2 = temp;
		}

		if (ty1 > ty2) {
			const float temp = ty1;
			ty1 = ty2;
			ty2 = temp;
		}

		if (tz1 > tz2) {
			const float temp = tz1;
			tz1 = tz2;
			tz2 = temp;
		}

		float t1 = tx1; if (t1 < ty1) t1 = ty1; if (t1 < tz1) t1 = tz1;
		float t2 = tx2; if (t2 > ty2) t2 = ty2; if (t2 > tz2) t2 = tz2;

		if (t1 > t2) return false;
		if ((t1 < 0.0) && (t2 < 0.0)) return false;

		minHit.t = t1;
		return true;
	}
};




// triangle
struct Triangle {
	float3 positions[3];
	float3 normals[3];
	float2 texcoords[3];
	int idMaterial = 0;
	AABB bbox;
	float3 center;
};

bool isPointInsideLine(float2 p, float2 p1, float2 p2, float2 p3) {
	float2 tangent = p2 - p1;
	float2 normal = {0-tangent.y, tangent.x};
	float2 v = {p.x - p1.x, p.y - p1.y};
	float insideLine = v.x * normal.x + v.y * normal.y;
	if (insideLine - Epsilon > 0) {
		return true;
	} else if (abs(insideLine) < Epsilon) {
		// return true if top edge or left edge
		return (tangent.y == 0 && p3.y < p1.y) || (tangent.y != 0 && normal.x < 0);
	} else {
		return false;
	}
}

// return true if p contained within the triangle formed by tpoints, false otherwise
bool isInsideTriangle(float2 p, const float2 tpoints[]) {
	bool l1 = isPointInsideLine(p, tpoints[0], tpoints[1], tpoints[2]);
	bool l2 = isPointInsideLine(p, tpoints[1], tpoints[2], tpoints[0]);
	bool l3 = isPointInsideLine(p, tpoints[2], tpoints[0], tpoints[1]);
	return (l1 && l2 && l3) || (!l1 && !l2 && !l3);
}

float triangleArea(const float3 points[]) {
	return 0.5f * 
		abs(points[0].x * (points[1].y - points[2].y) + points[1].x * (points[2].y - points[0].y) + points[2].x * (points[0].y - points[1].y));
}

float3 barycentricCoordinates(const float3 points[], float x, float y) {
	float a = triangleArea(points);
	float3 p = {x, y, 0.0f};
	float3 t0[3] = {p, points[1], points[2]};
	float3 t1[3] = {p, points[0], points[2]};
	float a0 = triangleArea(t0);
	float a1 = triangleArea(t1);
	float a2 = a - a0 - a1;
	return {(a0 / a), (a1 / a), (a2 / a)};
}

float det(float3 cola, float3 colb, float3 colc) {
	return dot(cross(cola, colb), colc);
}

HitInfo interpolateHitInfo(const Triangle &tri, const float3 baryCoords, float W[]) {
	float3 pcBaryCoords = {baryCoords[0] * W[0], baryCoords[1] * W[1], baryCoords[2] * W[2]};
	float interpolatedW = sum(pcBaryCoords);
	HitInfo hi;
	hi.T = (pcBaryCoords[0] * tri.texcoords[0] + 
			pcBaryCoords[1] * tri.texcoords[1] + 
			pcBaryCoords[2] * tri.texcoords[2]) / interpolatedW;
	hi.N = normalize(pcBaryCoords[0] * tri.normals[0] + pcBaryCoords[1] * tri.normals[1] + pcBaryCoords[2] * tri.normals[2]);
	hi.P = (pcBaryCoords[0] * tri.positions[0] + pcBaryCoords[1] * tri.positions[1] + pcBaryCoords[2] * tri.positions[2]) / interpolatedW;
	return hi;
}

// triangle mesh
static float3 shade(const HitInfo& hit, const float3& viewDir, const int level = 0);
class TriangleMesh {
public:
	std::vector<Triangle> triangles;
	std::vector<Material> materials;
	AABB bbox;

	void transform(const float4x4& m) {
		// ====== implement it if you want =====
		// matrix transformation of an object	
		// m is a matrix that transforms an object
		// implement proper transformation for positions and normals
		// (hint: you will need to have float4 versions of p and n)
		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			for (int k = 0; k <= 2; k++) {
				const float3 &p = this->triangles[i].positions[k];
				const float3 &n = this->triangles[i].normals[k];
				// not doing anything right now
			}
		}
	}

	void rasterizeTriangle(const Triangle& tri, const float4x4& plm) const {
		float3 screenPoints[3];
		float3 ndcPoints[3];
		float W[3];

		for (int k = 0; k < 3; k++) {
			// apply transformations
			float4 p = {tri.positions[k][0], tri.positions[k][1], tri.positions[k][2], 1.0f};
			p = mul(plm, p);

			// get NDC coordinates
			ndcPoints[k] = {p.x / p.w, p.y / p.w, p.z / p.w};

			// get screen coordinates
			screenPoints[k] = {
				(1.0f + ndcPoints[k].x) * FrameBuffer.width * 0.5f,
				(1.0f + ndcPoints[k].y) * FrameBuffer.height * 0.5f,
				ndcPoints[k].z
			};
			W[k] = 1.0f / p.w;
		}

		// iterate over all pixels (x,y), check if (x,y) is inside the triangle, shade it
		int smallestScreenX = (int)fmax(0.0f, fmin(fmin(screenPoints[0].x, screenPoints[1].x), screenPoints[2].x) - 2);
		int smallestScreenY = (int)fmax(0.0f, fmin(fmin(screenPoints[0].y, screenPoints[1].y), screenPoints[2].y) - 2);
		int largestScreenX = (int)fmin(FrameBuffer.width, fmax(fmax(screenPoints[0].x, screenPoints[1].x), screenPoints[2].x) + 2);
		int largestScreenY = (int)fmin(FrameBuffer.height, fmax(fmax(screenPoints[0].y, screenPoints[1].y), screenPoints[2].y) + 2);
		float2 screenPoints2d[3];
		for (int i = 0; i < 3; ++i) screenPoints2d[i] = {screenPoints[i].x, screenPoints[i].y};
		for (int x = smallestScreenX; x < largestScreenX; x++) {
			for (int y = smallestScreenY; y < largestScreenY; y++) {
				if (isInsideTriangle({x + 0.5f, y + 0.5f}, screenPoints2d)) {
					float ndcX = (x + 0.5f) * 2.0f / FrameBuffer.width - 1.0f;
					float ndcY = (y + 0.5f) * 2.0f / FrameBuffer.height - 1.0f;
					float3 baryCoords = barycentricCoordinates(ndcPoints, ndcX, ndcY);
					float ndcZ = dot(baryCoords, {ndcPoints[0].z, ndcPoints[1].z, ndcPoints[2].z});
					if (ndcZ < FrameBuffer.depth(x, y)) {
						HitInfo hi = interpolateHitInfo(tri, baryCoords, W);
						hi.material = &materials[tri.idMaterial];
						FrameBuffer.pixel(x, y) = shade(hi, globalViewDir);
						FrameBuffer.depth(x, y) = ndcZ;
					}
				}
			}
		}
	}


	bool raytraceTriangle(HitInfo& result, const Ray& ray, const Triangle& tri, float tMin, float tMax) const {
		// ====== implement it in A2 ======
		// ray-triangle intersection
		// fill in "result" when there is an intersection
		// return true/false if there is an intersection or not

		float3 a = tri.positions[0] - tri.positions[1];
		float3 b = tri.positions[0] - tri.positions[2];
		float3 c = ray.d;
		float3 d = tri.positions[0] - ray.o;

		float D = det(a, b, c);
		if (D != 0) {
			float Da = det(d, b, c);
			float Db = det(a, d, c);
			float Dc = det(a, b, d);
			float beta = Da / D;
			float gamma = Db / D;
			float alpha = 1 - beta - gamma;
			float t = Dc / D;
			if (
				0 <= alpha && alpha <= 1 &&
				0 <= beta && beta <= 1 &&
				0 <= gamma && gamma <= 1 &&
				tMin <= t && t <= tMax
			) {
				result.material = &materials[tri.idMaterial];
				result.T = alpha * tri.texcoords[0] + beta * tri.texcoords[1] + gamma * tri.texcoords[2];
				result.t = t;
				result.N = normalize(alpha * tri.normals[0] + beta * tri.normals[1] + gamma * tri.normals[2]);
				result.P = alpha * tri.positions[0] + beta * tri.positions[1] + gamma * tri.positions[2];
				return true;
			}
		}

		return false;
	}


	// some precalculation for bounding boxes (you do not need to change it)
	void preCalc() {
		bbox.reset();
		for (int i = 0, _n = (int)triangles.size(); i < _n; i++) {
			this->triangles[i].bbox.reset();
			this->triangles[i].bbox.fit(this->triangles[i].positions[0]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[1]);
			this->triangles[i].bbox.fit(this->triangles[i].positions[2]);

			this->triangles[i].center = (this->triangles[i].positions[0] + this->triangles[i].positions[1] + this->triangles[i].positions[2]) * (1.0f / 3.0f);

			this->bbox.fit(this->triangles[i].positions[0]);
			this->bbox.fit(this->triangles[i].positions[1]);
			this->bbox.fit(this->triangles[i].positions[2]);
		}
	}


	// load .obj file (you do not need to modify it unless you want to change something)
	bool load(const char* filename, const float4x4& ctm = linalg::identity) {
		int nVertices = 0;
		float* vertices;
		float* normals;
		float* texcoords;
		int nIndices;
		int* indices;
		int* matid = nullptr;

		printf("Loading \"%s\"...\n", filename);
		ParseOBJ(filename, nVertices, &vertices, &normals, &texcoords, nIndices, &indices, &matid);
		if (nVertices == 0) return false;
		this->triangles.resize(nIndices / 3);

		if (matid != nullptr) {
			for (unsigned int i = 0; i < materials.size(); i++) {
				// convert .mlt data into BSDF definitions
				// you may change the followings in the final project if you want
				materials[i].type = MAT_LAMBERTIAN;
				if (materials[i].Ns == 100.0f) {
					materials[i].type = MAT_METAL;
				}
				if (materials[i].name.compare(0, 5, "glass", 0, 5) == 0) {
					materials[i].type = MAT_GLASS;
					materials[i].eta = 1.5f;
				}
			}
		} else {
			// use default Lambertian
			this->materials.resize(1);
		}

		for (unsigned int i = 0; i < this->triangles.size(); i++) {
			const int v0 = indices[i * 3 + 0];
			const int v1 = indices[i * 3 + 1];
			const int v2 = indices[i * 3 + 2];

			this->triangles[i].positions[0] = float3(vertices[v0 * 3 + 0], vertices[v0 * 3 + 1], vertices[v0 * 3 + 2]);
			this->triangles[i].positions[1] = float3(vertices[v1 * 3 + 0], vertices[v1 * 3 + 1], vertices[v1 * 3 + 2]);
			this->triangles[i].positions[2] = float3(vertices[v2 * 3 + 0], vertices[v2 * 3 + 1], vertices[v2 * 3 + 2]);

			if (normals != nullptr) {
				this->triangles[i].normals[0] = float3(normals[v0 * 3 + 0], normals[v0 * 3 + 1], normals[v0 * 3 + 2]);
				this->triangles[i].normals[1] = float3(normals[v1 * 3 + 0], normals[v1 * 3 + 1], normals[v1 * 3 + 2]);
				this->triangles[i].normals[2] = float3(normals[v2 * 3 + 0], normals[v2 * 3 + 1], normals[v2 * 3 + 2]);
			} else {
				// no normal data, calculate the normal for a polygon
				const float3 e0 = this->triangles[i].positions[1] - this->triangles[i].positions[0];
				const float3 e1 = this->triangles[i].positions[2] - this->triangles[i].positions[0];
				const float3 n = normalize(cross(e0, e1));

				this->triangles[i].normals[0] = n;
				this->triangles[i].normals[1] = n;
				this->triangles[i].normals[2] = n;
			}

			// material id
			this->triangles[i].idMaterial = 0;
			if (matid != nullptr) {
				// read texture coordinates
				if ((texcoords != nullptr) && materials[matid[i]].isTextured) {
					this->triangles[i].texcoords[0] = float2(texcoords[v0 * 2 + 0], texcoords[v0 * 2 + 1]);
					this->triangles[i].texcoords[1] = float2(texcoords[v1 * 2 + 0], texcoords[v1 * 2 + 1]);
					this->triangles[i].texcoords[2] = float2(texcoords[v2 * 2 + 0], texcoords[v2 * 2 + 1]);
				} else {
					this->triangles[i].texcoords[0] = float2(0.0f);
					this->triangles[i].texcoords[1] = float2(0.0f);
					this->triangles[i].texcoords[2] = float2(0.0f);
				}
				this->triangles[i].idMaterial = matid[i];
			} else {
				this->triangles[i].texcoords[0] = float2(0.0f);
				this->triangles[i].texcoords[1] = float2(0.0f);
				this->triangles[i].texcoords[2] = float2(0.0f);
			}
		}
		printf("Loaded \"%s\" with %d triangles.\n", filename, int(triangles.size()));

		delete[] vertices;
		delete[] normals;
		delete[] texcoords;
		delete[] indices;
		delete[] matid;

		return true;
	}

	~TriangleMesh() {
		materials.clear();
		triangles.clear();
	}


	bool bruteforceIntersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) {
		// bruteforce ray tracing (for debugging)
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		for (int i = 0; i < triangles.size(); ++i) {
			if (raytraceTriangle(tempMinHit, ray, triangles[i], tMin, tMax)) {
				if (tempMinHit.t < result.t) {
					hit = true;
					result = tempMinHit;
				}
			}
		}

		return hit;
	}

	void createSingleTriangle() {
		triangles.resize(1);
		materials.resize(1);

		triangles[0].idMaterial = 0;

		triangles[0].positions[0] = float3(-0.5f, -0.5f, 0.0f);
		triangles[0].positions[1] = float3(0.5f, -0.5f, 0.0f);
		triangles[0].positions[2] = float3(0.0f, 0.5f, 0.0f);

		const float3 e0 = this->triangles[0].positions[1] - this->triangles[0].positions[0];
		const float3 e1 = this->triangles[0].positions[2] - this->triangles[0].positions[0];
		const float3 n = normalize(cross(e0, e1));

		triangles[0].normals[0] = n;
		triangles[0].normals[1] = n;
		triangles[0].normals[2] = n;

		triangles[0].texcoords[0] = float2(0.0f, 0.0f);
		triangles[0].texcoords[1] = float2(0.0f, 1.0f);
		triangles[0].texcoords[2] = float2(1.0f, 0.0f);
	}


private:
	// === you do not need to modify the followings in this class ===
	void loadTexture(const char* fname, const int i) {
		int comp;
		materials[i].texture = stbi_load(fname, &materials[i].textureWidth, &materials[i].textureHeight, &comp, 3);
		if (!materials[i].texture) {
			std::cerr << "Unable to load texture: " << fname << std::endl;
			return;
		}
	}

	std::string GetBaseDir(const std::string& filepath) {
		if (filepath.find_last_of("/\\") != std::string::npos) return filepath.substr(0, filepath.find_last_of("/\\"));
		return "";
	}
	std::string base_dir;

	void LoadMTL(const std::string fileName) {
		FILE* fp = fopen(fileName.c_str(), "r");

		Material mtl;
		mtl.texture = nullptr;
		char line[81];
		while (fgets(line, 80, fp) != nullptr) {
			float r, g, b, s;
			std::string lineStr;
			lineStr = line;
			int i = int(materials.size());

			if (lineStr.compare(0, 6, "newmtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				mtl.name = lineStr;
				mtl.isTextured = false;
			} else if (lineStr.compare(0, 2, "Ka", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ka = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Kd", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Kd = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ks", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f %f %f\n", &r, &g, &b);
				mtl.Ks = float3(r, g, b);
			} else if (lineStr.compare(0, 2, "Ns", 0, 2) == 0) {
				lineStr.erase(0, 3);
				sscanf(lineStr.c_str(), "%f\n", &s);
				mtl.Ns = s;
				mtl.texture = nullptr;
				materials.push_back(mtl);
			} else if (lineStr.compare(0, 6, "map_Kd", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				materials[i - 1].isTextured = true;
				loadTexture((base_dir + lineStr).c_str(), i - 1);
			}
		}

		fclose(fp);
	}

	void ParseOBJ(const char* fileName, int& nVertices, float** vertices, float** normals, float** texcoords, int& nIndices, int** indices, int** materialids) {
		// local function in C++...
		struct {
			void operator()(char* word, int* vindex, int* tindex, int* nindex) {
				const char* null = " ";
				char* ptr;
				const char* tp;
				const char* np;

				// by default, the texture and normal pointers are set to the null string
				tp = null;
				np = null;

				// replace slashes with null characters and cause tp and np to point
				// to character immediately following the first or second slash
				for (ptr = word; *ptr != '\0'; ptr++) {
					if (*ptr == '/') {
						if (tp == null) {
							tp = ptr + 1;
						} else {
							np = ptr + 1;
						}

						*ptr = '\0';
					}
				}

				*vindex = atoi(word);
				*tindex = atoi(tp);
				*nindex = atoi(np);
			}
		} get_indices;

		base_dir = GetBaseDir(fileName);
		#ifdef _WIN32
			base_dir += "\\";
		#else
			base_dir += "/";
		#endif

		FILE* fp = fopen(fileName, "r");
		int nv = 0, nn = 0, nf = 0, nt = 0;
		char line[81];
		if (!fp) {
			printf("Cannot open \"%s\" for reading\n", fileName);
			return;
		}

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (lineStr.compare(0, 6, "mtllib", 0, 6) == 0) {
				lineStr.erase(0, 7);
				lineStr.erase(lineStr.size() - 1, 1);
				LoadMTL(base_dir + lineStr);
			}

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					nn++;
				} else if (line[1] == 't') {
					nt++;
				} else {
					nv++;
				}
			} else if (line[0] == 'f') {
				nf++;
			}
		}
		fseek(fp, 0, 0);

		float* n = new float[3 * (nn > nf ? nn : nf)];
		float* v = new float[3 * nv];
		float* t = new float[2 * nt];

		int* vInd = new int[3 * nf];
		int* nInd = new int[3 * nf];
		int* tInd = new int[3 * nf];
		int* mInd = new int[nf];

		int nvertices = 0;
		int nnormals = 0;
		int ntexcoords = 0;
		int nindices = 0;
		int ntriangles = 0;
		bool noNormals = false;
		bool noTexCoords = false;
		bool noMaterials = true;
		int cmaterial = 0;

		while (fgets(line, 80, fp) != NULL) {
			std::string lineStr;
			lineStr = line;

			if (line[0] == 'v') {
				if (line[1] == 'n') {
					float x, y, z;
					sscanf(&line[2], "%f %f %f\n", &x, &y, &z);
					float l = sqrt(x * x + y * y + z * z);
					x = x / l;
					y = y / l;
					z = z / l;
					n[nnormals] = x;
					nnormals++;
					n[nnormals] = y;
					nnormals++;
					n[nnormals] = z;
					nnormals++;
				} else if (line[1] == 't') {
					float u, v;
					sscanf(&line[2], "%f %f\n", &u, &v);
					t[ntexcoords] = u;
					ntexcoords++;
					t[ntexcoords] = v;
					ntexcoords++;
				} else {
					float x, y, z;
					sscanf(&line[1], "%f %f %f\n", &x, &y, &z);
					v[nvertices] = x;
					nvertices++;
					v[nvertices] = y;
					nvertices++;
					v[nvertices] = z;
					nvertices++;
				}
			}
			if (lineStr.compare(0, 6, "usemtl", 0, 6) == 0) {
				lineStr.erase(0, 7);
				if (materials.size() != 0) {
					for (unsigned int i = 0; i < materials.size(); i++) {
						if (lineStr.compare(materials[i].name) == 0) {
							cmaterial = i;
							noMaterials = false;
							break;
						}
					}
				}

			} else if (line[0] == 'f') {
				char s1[32], s2[32], s3[32];
				int vI, tI, nI;
				sscanf(&line[1], "%s %s %s\n", s1, s2, s3);

				mInd[ntriangles] = cmaterial;

				// indices for first vertex
				get_indices(s1, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for second vertex
				get_indices(s2, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				// indices for third vertex
				get_indices(s3, &vI, &tI, &nI);
				vInd[nindices] = vI - 1;
				if (nI) {
					nInd[nindices] = nI - 1;
				} else {
					noNormals = true;
				}

				if (tI) {
					tInd[nindices] = tI - 1;
				} else {
					noTexCoords = true;
				}
				nindices++;

				ntriangles++;
			}
		}

		*vertices = new float[ntriangles * 9];
		if (!noNormals) {
			*normals = new float[ntriangles * 9];
		} else {
			*normals = 0;
		}

		if (!noTexCoords) {
			*texcoords = new float[ntriangles * 6];
		} else {
			*texcoords = 0;
		}

		if (!noMaterials) {
			*materialids = new int[ntriangles];
		} else {
			*materialids = 0;
		}

		*indices = new int[ntriangles * 3];
		nVertices = ntriangles * 3;
		nIndices = ntriangles * 3;

		for (int i = 0; i < ntriangles; i++) {
			if (!noMaterials) {
				(*materialids)[i] = mInd[i];
			}

			(*indices)[3 * i] = 3 * i;
			(*indices)[3 * i + 1] = 3 * i + 1;
			(*indices)[3 * i + 2] = 3 * i + 2;

			(*vertices)[9 * i] = v[3 * vInd[3 * i]];
			(*vertices)[9 * i + 1] = v[3 * vInd[3 * i] + 1];
			(*vertices)[9 * i + 2] = v[3 * vInd[3 * i] + 2];

			(*vertices)[9 * i + 3] = v[3 * vInd[3 * i + 1]];
			(*vertices)[9 * i + 4] = v[3 * vInd[3 * i + 1] + 1];
			(*vertices)[9 * i + 5] = v[3 * vInd[3 * i + 1] + 2];

			(*vertices)[9 * i + 6] = v[3 * vInd[3 * i + 2]];
			(*vertices)[9 * i + 7] = v[3 * vInd[3 * i + 2] + 1];
			(*vertices)[9 * i + 8] = v[3 * vInd[3 * i + 2] + 2];

			if (!noNormals) {
				(*normals)[9 * i] = n[3 * nInd[3 * i]];
				(*normals)[9 * i + 1] = n[3 * nInd[3 * i] + 1];
				(*normals)[9 * i + 2] = n[3 * nInd[3 * i] + 2];

				(*normals)[9 * i + 3] = n[3 * nInd[3 * i + 1]];
				(*normals)[9 * i + 4] = n[3 * nInd[3 * i + 1] + 1];
				(*normals)[9 * i + 5] = n[3 * nInd[3 * i + 1] + 2];

				(*normals)[9 * i + 6] = n[3 * nInd[3 * i + 2]];
				(*normals)[9 * i + 7] = n[3 * nInd[3 * i + 2] + 1];
				(*normals)[9 * i + 8] = n[3 * nInd[3 * i + 2] + 2];
			}

			if (!noTexCoords) {
				(*texcoords)[6 * i] = t[2 * tInd[3 * i]];
				(*texcoords)[6 * i + 1] = t[2 * tInd[3 * i] + 1];

				(*texcoords)[6 * i + 2] = t[2 * tInd[3 * i + 1]];
				(*texcoords)[6 * i + 3] = t[2 * tInd[3 * i + 1] + 1];

				(*texcoords)[6 * i + 4] = t[2 * tInd[3 * i + 2]];
				(*texcoords)[6 * i + 5] = t[2 * tInd[3 * i + 2] + 1];
			}

		}
		fclose(fp);

		delete[] n;
		delete[] v;
		delete[] t;
		delete[] nInd;
		delete[] vInd;
		delete[] tInd;
		delete[] mInd;
	}
};

class BVHNode {
public:
	bool isLeaf;
	int idLeft, idRight;
	int triListNum;
	int* triList;
	AABB bbox;
};

class BVH {
public:
	const TriangleMesh* triangleMesh = nullptr;
	BVHNode* node = nullptr;

	const float costBBox = 1.0f;
	const float costTri = 1.0f;

	int leafNum = 0;
	int nodeNum = 0;

	BVH() {}
	void build(const TriangleMesh* mesh);

	bool intersect(HitInfo& result, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		result.t = FLT_MAX;

		// bvh
		if (this->node[0].bbox.intersect(tempMinHit, ray)) {
			hit = traverse(result, ray, 0, tMin, tMax);
		}
		if (result.t != FLT_MAX) hit = true;

		return hit;
	}
	bool traverse(HitInfo& result, const Ray& ray, int node_id, float tMin, float tMax) const;

private:
	void sortAxis(int* obj_index, const char axis, const int li, const int ri) const;
	int splitBVH(int* obj_index, const int obj_num, const AABB& bbox);

};


// sort bounding boxes (in case you want to build SAH-BVH)
void BVH::sortAxis(int* obj_index, const char axis, const int li, const int ri) const {
	int i, j;
	float pivot;
	int temp;

	i = li;
	j = ri;

	pivot = triangleMesh->triangles[obj_index[(li + ri) / 2]].center[axis];

	while (true) {
		while (triangleMesh->triangles[obj_index[i]].center[axis] < pivot) {
			++i;
		}

		while (triangleMesh->triangles[obj_index[j]].center[axis] > pivot) {
			--j;
		}

		if (i >= j) break;

		temp = obj_index[i];
		obj_index[i] = obj_index[j];
		obj_index[j] = temp;

		++i;
		--j;
	}

	if (li < (i - 1)) sortAxis(obj_index, axis, li, i - 1);
	if ((j + 1) < ri) sortAxis(obj_index, axis, j + 1, ri);
}


#define SAHBVH // use this in once you have SAH-BVH
int BVH::splitBVH(int* obj_index, const int obj_num, const AABB& bbox) {
	// ====== extend it in A2 extra ======
	int bestAxis, bestIndex;
	AABB bboxL, bboxR, bestbboxL, bestbboxR;
	int* sorted_obj_index = new int[obj_num];
	bool shouldSplit;

#ifndef SAHBVH
	// split along the largest axis
	bestAxis = bbox.getLargestAxis();

	// sorting along the axis
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[i] = obj_index[i];
	}

	// split in the middle
	bestIndex = obj_num / 2 - 1;

	// split until 4 or less objects
	shouldSplit = (obj_num <= 4);
#else
	float outerArea = bbox.area();
	float cmin = INFINITY;

	float* bboxAreasL = new float[obj_num];
	float* bboxAreasR = new float[obj_num];

	// iterate over each axis to find best one
	for (int axis = 0; axis <= 2; ++axis) {
		
		// sorting along the axis
		this->sortAxis(obj_index, axis, 0, obj_num - 1);
		for (int i = 0; i < obj_num; ++i) {
			sorted_obj_index[i] = obj_index[i];
		}

		// find surface areas of all possible bboxes used by greedy
		bboxL.reset();
		bboxR.reset();
		for (int split_idx = 0; split_idx < obj_num; ++split_idx) {
			int li = sorted_obj_index[split_idx];
			int ri = sorted_obj_index[obj_num - split_idx];
			const Triangle& leftTri = triangleMesh->triangles[li];
			bboxL.fit(leftTri.positions[0]);
			bboxL.fit(leftTri.positions[1]);
			bboxL.fit(leftTri.positions[2]);
			if (split_idx > 0) {
				const Triangle& rightTri = triangleMesh->triangles[ri];
				bboxR.fit(rightTri.positions[0]);
				bboxR.fit(rightTri.positions[1]);
				bboxR.fit(rightTri.positions[2]);
			}
			bboxAreasL[split_idx] = bboxL.area();
			bboxAreasR[obj_num - split_idx - 1] = bboxR.area();
		}

		// find best index to split on using SAH
		for (int i = 0; i < obj_num; ++i) {
			// 2Cb + (Area(left) / Area(parent)) * 2C0 + (Area(right) / Area(parent)) * 2C0
			float c = 2 * Cb + (bboxAreasL[i] * i + bboxAreasR[i] * (obj_num - i)) * C0 / outerArea;
			if (c < cmin) {
				cmin = c;
				bestIndex = i;
				bestAxis = axis;
			}
		}
	}

	// sorting along the axis
	this->sortAxis(obj_index, bestAxis, 0, obj_num - 1);
	for (int i = 0; i < obj_num; ++i) {
		sorted_obj_index[i] = obj_index[i];
	}

	// split only when it is better to split
	shouldSplit = (obj_num <= 4 || cmin > obj_num);

	delete [] bboxAreasL;
	delete [] bboxAreasR;
#endif

	if (shouldSplit) {
		delete[] sorted_obj_index;

		this->nodeNum++;
		this->node[this->nodeNum - 1].bbox = bbox;
		this->node[this->nodeNum - 1].isLeaf = true;
		this->node[this->nodeNum - 1].triListNum = obj_num;
		this->node[this->nodeNum - 1].triList = new int[obj_num];
		for (int i = 0; i < obj_num; i++) {
			this->node[this->nodeNum - 1].triList[i] = obj_index[i];
		}
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->leafNum++;

		return temp_id;
	} else {
		// use bestIndex to construct bestbboxes
		bestbboxL.reset();
		for (int i = 0; i <= bestIndex; ++i) {
			const Triangle& tri = triangleMesh->triangles[sorted_obj_index[i]];
			bestbboxL.fit(tri.positions[0]);
			bestbboxL.fit(tri.positions[1]);
			bestbboxL.fit(tri.positions[2]);
		}

		bestbboxR.reset();
		for (int i = bestIndex + 1; i < obj_num; ++i) {
			const Triangle& tri = triangleMesh->triangles[sorted_obj_index[i]];
			bestbboxR.fit(tri.positions[0]);
			bestbboxR.fit(tri.positions[1]);
			bestbboxR.fit(tri.positions[2]);
		}

		// split obj_index into two 
		int* obj_indexL = new int[bestIndex + 1];
		int* obj_indexR = new int[obj_num - (bestIndex + 1)];
		for (int i = 0; i <= bestIndex; ++i) {
			obj_indexL[i] = sorted_obj_index[i];
		}
		for (int i = bestIndex + 1; i < obj_num; ++i) {
			obj_indexR[i - (bestIndex + 1)] = sorted_obj_index[i];
		}
		delete[] sorted_obj_index;
		int obj_numL = bestIndex + 1;
		int obj_numR = obj_num - (bestIndex + 1);

		// recursive call to build a tree
		this->nodeNum++;
		int temp_id;
		temp_id = this->nodeNum - 1;
		this->node[temp_id].bbox = bbox;
		this->node[temp_id].isLeaf = false;
		this->node[temp_id].idLeft = splitBVH(obj_indexL, obj_numL, bestbboxL);
		this->node[temp_id].idRight = splitBVH(obj_indexR, obj_numR, bestbboxR);

		delete[] obj_indexL;
		delete[] obj_indexR;

		return temp_id;
	}
}


// you may keep this part as-is
void BVH::build(const TriangleMesh* mesh) {
	triangleMesh = mesh;

	// construct the bounding volume hierarchy
	const int obj_num = (int)(triangleMesh->triangles.size());
	int* obj_index = new int[obj_num];
	for (int i = 0; i < obj_num; ++i) {
		obj_index[i] = i;
	}
	this->nodeNum = 0;
	this->node = new BVHNode[obj_num * 2];
	this->leafNum = 0;

	// calculate a scene bounding box
	AABB bbox;
	for (int i = 0; i < obj_num; i++) {
		const Triangle& tri = triangleMesh->triangles[obj_index[i]];

		bbox.fit(tri.positions[0]);
		bbox.fit(tri.positions[1]);
		bbox.fit(tri.positions[2]);
	}

	// ---------- buliding BVH ----------
	printf("Building BVH...\n");
	splitBVH(obj_index, obj_num, bbox);
	printf("Done.\n");

	delete[] obj_index;
}


// you may keep this part as-is
bool BVH::traverse(HitInfo& minHit, const Ray& ray, int node_id, float tMin, float tMax) const {
	bool hit = false;
	HitInfo tempMinHit, tempMinHitL, tempMinHitR;
	bool hit1, hit2;

	if (this->node[node_id].isLeaf) {
		for (int i = 0; i < (this->node[node_id].triListNum); ++i) {
			if (triangleMesh->raytraceTriangle(tempMinHit, ray, triangleMesh->triangles[this->node[node_id].triList[i]], tMin, tMax)) {
				hit = true;
				if (tempMinHit.t < minHit.t) minHit = tempMinHit;
			}
		}
	} else {
		hit1 = this->node[this->node[node_id].idLeft].bbox.intersect(tempMinHitL, ray);
		hit2 = this->node[this->node[node_id].idRight].bbox.intersect(tempMinHitR, ray);

		hit1 = hit1 && (tempMinHitL.t < minHit.t);
		hit2 = hit2 && (tempMinHitR.t < minHit.t);

		if (hit1 && hit2) {
			if (tempMinHitL.t < tempMinHitR.t) {
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
			} else {
				hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
				hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
			}
		} else if (hit1) {
			hit = traverse(minHit, ray, this->node[node_id].idLeft, tMin, tMax);
		} else if (hit2) {
			hit = traverse(minHit, ray, this->node[node_id].idRight, tMin, tMax);
		}
	}

	return hit;
}

// ====== implement it in A3 ======
// fill in the missing parts
class Particle {
public:
	float3 position = float3(0.0f);
	float3 velocity = float3(0.0f);
	float3 prevPosition = position;
	float mass = 4.0e9f;

	void reset() {
		position = float3(PCG32::rand(), PCG32::rand(), PCG32::rand()) - float(0.5f);
		velocity = 2.0f * float3((PCG32::rand() - 0.5f), 0.0f, (PCG32::rand() - 0.5f));
		prevPosition = position;
		position += velocity * deltaT;
	}

	void step(float3 force) {
		float3 temp = position;

		// === fill in this part in A3 ===
		// TASK 1
		// update the particle position and velocity here
		// position = position + (position - prevPosition) + powf(deltaT, 2) * globalGravity;
		// TASK 4
		position = position + (position - prevPosition) + powf(deltaT, 2) * (globalGravity + force * (1.0f / mass));
		prevPosition = temp;

		// TASK 2
		// perform collisions on [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5] box
		// for (int i = 0; i < 3; ++i) {
		// 	if (position[i] < -0.5f) {
		// 		prevPosition[i] = 2 * -0.5f - prevPosition[i];
		// 		prevPosition[i] += (-0.5f - position[i]);
		// 		position[i] = -0.5f;
		// 	} else if (position[i] > 0.5f) {
		// 		prevPosition[i] = 2 * 0.5f - prevPosition[i];
		// 		prevPosition[i] += (0.5f - position[i]);
		// 		position[i] = 0.5f;
		// 	}
		// }

		// TASK 3
		// perform collisions on a sphere of radius 1 centered at the origin
		// since r = 1 and c = 0, the formula for projecting position onto the sphere becomes position / ||position||
		position = normalize(position);
	}
};

struct OctreeNode {
    float3 center;        // Center of this node
    float size;           // Size of this node
    float3 massCenter;    // Center of mass of particles in this node
    int mass;             // Total mass of particles in this node (given as the number of particles since mass is same for all particles)
    std::vector<int> particles;
    std::vector<OctreeNode*> children;
};

class ParticleSystem {
public:
	std::vector<Particle> particles;
	TriangleMesh particlesMesh;
	TriangleMesh sphere;
	const char* sphereMeshFilePath = 0;
	float sphereSize = 0.0f;
	ParticleSystem() {};

	void updateMesh() {
		// you can optionally update the other mesh information (e.g., bounding box, BVH - which is tricky)
		if (sphereSize > 0) {
			const int n = int(sphere.triangles.size());
			for (int i = 0; i < globalNumParticles; i++) {
				for (int j = 0; j < n; j++) {
					particlesMesh.triangles[i * n + j].positions[0] = sphere.triangles[j].positions[0] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[1] = sphere.triangles[j].positions[1] + particles[i].position;
					particlesMesh.triangles[i * n + j].positions[2] = sphere.triangles[j].positions[2] + particles[i].position;
					particlesMesh.triangles[i * n + j].normals[0] = sphere.triangles[j].normals[0];
					particlesMesh.triangles[i * n + j].normals[1] = sphere.triangles[j].normals[1];
					particlesMesh.triangles[i * n + j].normals[2] = sphere.triangles[j].normals[2];
				}
			}
		} else {
			const float particleSize = 0.005f;
			for (int i = 0; i < globalNumParticles; i++) {
				// facing toward the camera
				particlesMesh.triangles[i].positions[0] = particles[i].position;
				particlesMesh.triangles[i].positions[1] = particles[i].position + particleSize * globalUp;
				particlesMesh.triangles[i].positions[2] = particles[i].position + particleSize * globalRight;
				particlesMesh.triangles[i].normals[0] = -globalViewDir;
				particlesMesh.triangles[i].normals[1] = -globalViewDir;
				particlesMesh.triangles[i].normals[2] = -globalViewDir;
			}
		}
	}

	void initialize() {
		particles.resize(globalNumParticles);
		particlesMesh.materials.resize(1);
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].reset();
		}

		if (sphereMeshFilePath) {
			if (sphere.load(sphereMeshFilePath)) {
				particlesMesh.triangles.resize(sphere.triangles.size() * globalNumParticles);
				sphere.preCalc();
				sphereSize = sphere.bbox.get_size().x * 0.5f;
			} else {
				particlesMesh.triangles.resize(globalNumParticles);
			}
		} else {
			particlesMesh.triangles.resize(globalNumParticles);
		}
		updateMesh();
	}

	void computeMassDistribution(OctreeNode* node) {
		if (node->particles.size() == 1) {
			node->mass = 1;
			node->massCenter = particles[node->particles[0]].position;
		} else if (node->children.size() > 0) {
			node->mass = 0;
			node->massCenter = float3{0.0f, 0.0f, 0.0f};
			for (auto child : node->children) {
				computeMassDistribution(child);
				if (child->mass > 0) {
					node->mass += child->mass;
					node->massCenter += child->massCenter * child->mass;
				}
			}
			node->massCenter = node->massCenter / node->mass;
		} else {
			node->mass = 0;
		}
	}

	void computeForce(int pId, OctreeNode* node, float3& accumulatedForce) {
		if (node->particles.size() == 1 && node->particles[0] == pId) return;
		Particle &p = particles[pId];
		float3 diff = node->massCenter - p.position;
		float dist = length(diff);
		if (node->size / dist < globalTheta || node->particles.size() == 1) {
			accumulatedForce += G * p.mass * p.mass * node->mass * diff / powf(dist + Epsilon, 3.0f);
		} else {
			for (auto& child : node->children) {
				computeForce(pId, child, accumulatedForce);
			}
		}
	}

	void insertParticle(OctreeNode* node, int pId) {
		// if empty leaf node, insert particle here
		if (node->particles.size() == 0 && node->children.empty()) {
			node->particles.push_back(pId);
			return;
		}

		// if non-empty leaf node, split into 8 new leaves
		if (node->particles.size() == 1 && node->children.empty()) {
			int existingParticle = node->particles[0];
			node->particles.clear();
			float halfSize = node->size / 2.0f;
			for (int i = 0; i < 8; ++i) {
				node->children.push_back(new OctreeNode());
				node->children[i]->center = node->center + float3{
					(i & 1 ? halfSize : -halfSize),
					(i & 2 ? halfSize : -halfSize),
					(i & 4 ? halfSize : -halfSize)};
				node->children[i]->size = halfSize;
			}
			insertParticle(node, existingParticle);
		}

		// Insert the particle into the correct child
		for (auto& child : node->children) {
			float px = particles[pId].position.x;
			float py = particles[pId].position.y;
			float pz = particles[pId].position.z;
			if (px >= child->center.x - child->size && px < child->center.x + child->size &&
				py >= child->center.y - child->size && py < child->center.y + child->size &&
				pz >= child->center.z - child->size && pz < child->center.z + child->size) {
				insertParticle(child, pId);
				return;
			}
		}
	}

	void computeAccumulatedForces(float3 accumulatedForces[]) {
	#ifdef A3_BONUS_1
		// build octree
		OctreeNode root;
		AABB box = AABB();
		for (auto &particle : particles) box.fit(particle.position);
		root.center = box.get_minp() + (box.get_maxp() - box.get_minp()) / 2;
		root.size = fmax(box.get_size().x, fmax(box.get_size().y, box.get_size().z));
		for (int i = 0; i < globalNumParticles; ++i) {
			insertParticle(&root, i);
		}

		// compute total masses and centre of masses for the octree
		computeMassDistribution(&root);

		// use octree to find total gravitational force acting on each particle
		for (int i = 0; i < globalNumParticles; ++i) {
			accumulatedForces[i] = float3(0.0f);
			computeForce(i, &root, accumulatedForces[i]);
		}
	#else
		for (int i = 0; i < globalNumParticles; ++i) {
			float3 f = float3(0.0f);
			for (int j = 0; j < globalNumParticles; ++j) {
				if (i == j) continue;
				float3 diff = particles[j].position - particles[i].position;
				float dist = sqrtf(powf(diff[0], 2) + powf(diff[1], 2) + powf(diff[2], 2));
				float3 forceij = G * particles[i].mass * particles[j].mass * diff / powf(dist + Epsilon, 3.0f);
				f += forceij;
			}
			accumulatedForces[i] = f;
		}
	#endif
	}

	void step() {
		// compute gravitational forces between particles
		float3 accumulatedForce[globalNumParticles];
		computeAccumulatedForces(accumulatedForce);

		// update particle positions
		for (int i = 0; i < globalNumParticles; i++) {
			particles[i].step(accumulatedForce[i]);
		}

		// collision detection
		for (int level = 0; level < 20; level++) {
			for (int i = 0; i < globalNumParticles; i++) {
				for (int j = i + 1; j < globalNumParticles; j++) {
					Particle &p1 = particles[i];
					Particle &p2 = particles[j];
					float dist = distance(p1.position, p2.position);
					float dp = 2 * sphereSize - dist;
					if (0 <= dp && dp > Epsilon) {
						// calculations here assume equal mass
						float3 delta = dp * 0.5f * normalize(p1.position - p2.position);
						float3 u1 = p1.position - p1.prevPosition;
						float3 u2 = p2.position - p2.prevPosition;
						float3 k = normalize(p1.position - p2.position);
						float3 a = k * (u1 - u2);
						float3 v1 = u1 - a * k;
						float3 v2 = u2 + a * k;
						p1.position += delta;
						p1.prevPosition = p1.position - v1;
						p2.position -= delta;
						p2.prevPosition = p2.position - v2;
					}
				}
			}
		}

		updateMesh();
	}
};
static ParticleSystem globalParticleSystem;

// scene definition
class Scene {
public:
	std::vector<TriangleMesh*> objects;
	std::vector<PointLightSource*> pointLightSources;
	std::vector<BVH> bvhs;
	Image envImage;

	float3 ibl(const Ray& ray) {
		if (envImage.height == 0) {
			return float3(0.0f);
		}
		float r = (1.0f / PI) * acos(ray.d.z) / sqrtf(pow(ray.d.x, 2) + pow(ray.d.y, 2));
		float u = ray.d.x * r;
		float v = ray.d.y * r;
		int i = (int)((u + 1.0f) * 0.5f * envImage.width);
		int j = (int)((v + 1.0f) * 0.5f * envImage.height);
		return envImage.pixel(i, j);
	}

	void addObject(TriangleMesh* pObj) {
		objects.push_back(pObj);
	}
	void addLight(PointLightSource* pObj) {
		pointLightSources.push_back(pObj);
	}

	void preCalc() {
		bvhs.resize(objects.size());
		for (int i = 0; i < objects.size(); i++) {
			objects[i]->preCalc();
			bvhs[i].build(objects[i]);
		}
	}

	// ray-scene intersection
	bool intersect(HitInfo& minHit, const Ray& ray, float tMin = 0.0f, float tMax = FLT_MAX) const {
		bool hit = false;
		HitInfo tempMinHit;
		minHit.t = FLT_MAX;

		for (int i = 0, i_n = (int)objects.size(); i < i_n; i++) {
			//if (objects[i]->bruteforceIntersect(tempMinHit, ray, tMin, tMax)) { // for debugging
			if (bvhs[i].intersect(tempMinHit, ray, tMin, tMax)) {
				if (tempMinHit.t < minHit.t) {
					hit = true;
					minHit = tempMinHit;
				}
			}
		}
		return hit;
	}

	// camera -> screen matrix (given to you for A1)
	float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar) const {
		float4x4 m;
		const float f = 1.0f / (tan(fovy * DegToRad / 2.0f));
		m[0] = { f / aspect, 0.0f, 0.0f, 0.0f };
		m[1] = { 0.0f, f, 0.0f, 0.0f };
		m[2] = { 0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), -1.0f };
		m[3] = { 0.0f, 0.0f, (2.0f * zFar * zNear) / (zNear - zFar), 0.0f };

		return m;
	}

	// model -> camera matrix (given to you for A1)
	float4x4 lookatMatrix(const float3& _eye, const float3& _center, const float3& _up) const {
		// transformation to the camera coordinate
		float4x4 m;
		const float3 f = normalize(_center - _eye);
		const float3 upp = normalize(_up);
		const float3 s = normalize(cross(f, upp));
		const float3 u = cross(s, f);

		m[0] = { s.x, s.y, s.z, 0.0f };
		m[1] = { u.x, u.y, u.z, 0.0f };
		m[2] = { -f.x, -f.y, -f.z, 0.0f };
		m[3] = { 0.0f, 0.0f, 0.0f, 1.0f };
		m = transpose(m);

		// translation according to the camera location
		const float4x4 t = float4x4{ {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}, { -_eye.x, -_eye.y, -_eye.z, 1.0f} };

		m = mul(m, t);
		return m;
	}

	// rasterizer
	void Rasterize() const {
		// ====== implement it in A1 ======
		// fill in plm by a proper matrix
		const float4x4 pm = perspectiveMatrix(globalFOV, globalAspectRatio, globalDepthMin, globalDepthMax);
		const float4x4 lm = lookatMatrix(globalEye, globalLookat, globalUp);
		const float4x4 plm = mul(pm, lm);

		FrameBuffer.clear();
		for (int n = 0, n_n = (int)objects.size(); n < n_n; n++) {
			for (int k = 0, k_n = (int)objects[n]->triangles.size(); k < k_n; k++) {
				objects[n]->rasterizeTriangle(objects[n]->triangles[k], plm);
			}
		}
	}

	// eye ray generation (given to you for A2)
	Ray eyeRay(int x, int y) const {
		// compute the camera coordinate system 
		const float3 wDir = normalize(float3(-globalViewDir));
		const float3 uDir = normalize(cross(globalUp, wDir));
		const float3 vDir = cross(wDir, uDir);

		// compute the pixel location in the world coordinate system using the camera coordinate system
		// trace a ray through the center of each pixel
		const float imPlaneUPos = (x + 0.5f) / float(globalWidth) - 0.5f;
		const float imPlaneVPos = (y + 0.5f) / float(globalHeight) - 0.5f;

		const float3 pixelPos = globalEye + float(globalAspectRatio * globalFilmSize * imPlaneUPos) * uDir + float(globalFilmSize * imPlaneVPos) * vDir - globalDistanceToFilm * wDir;

		return Ray(globalEye, normalize(pixelPos - globalEye));
	}

	// ray tracing (you probably don't need to change it in A2)
	void Raytrace() {
		FrameBuffer.clear();

		// loop over all pixels in the image
		for (int j = 0; j < globalHeight; ++j) {
			for (int i = 0; i < globalWidth; ++i) {
				const Ray ray = eyeRay(i, j);
				HitInfo hitInfo;
				if (intersect(hitInfo, ray)) {
					FrameBuffer.pixel(i, j) = shade(hitInfo, -ray.d);
				} else {
					FrameBuffer.pixel(i, j) = this->ibl(ray);
				}
			}

			// show intermediate process
			if (globalShowRaytraceProgress) {
				constexpr int scanlineNum = 64;
				if ((j % scanlineNum) == (scanlineNum - 1)) {
					glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0]);
					glRecti(1, 1, -1, -1);
					glfwSwapBuffers(globalGLFWindow);
					printf("Rendering Progress: %.3f%%\r", j / float(globalHeight - 1) * 100.0f);
					fflush(stdout);
				}
			}
		}
	}

	Ray generateRay(int x, int y) const {
		// compute the camera coordinate system 
		const float3 wDir = normalize(float3(-globalViewDir));
		const float3 uDir = normalize(cross(globalUp, wDir));
		const float3 vDir = cross(wDir, uDir);

		// compute the pixel location in the world coordinate system using the camera coordinate system
		// trace a ray through the center of each pixel with a random offset
		const float imPlaneUPos = (x + (PCG32::rand() * 2 - 1)) / float(globalWidth) - 0.5f;
		const float imPlaneVPos = (y + (PCG32::rand() * 2 - 1)) / float(globalHeight) - 0.5f;

		const float3 pixelPos = globalEye + float(globalAspectRatio * globalFilmSize * imPlaneUPos) * uDir + float(globalFilmSize * imPlaneVPos) * vDir - globalDistanceToFilm * wDir;

		return Ray(globalEye, normalize(pixelPos - globalEye));
	}

	void pathtraceSegment(int xmin, int xmax) {
		// loop over all pixels in the image
		for (int j = 0; j < globalHeight; ++j) {
			for (int i = xmin; i <= xmax; ++i) {
				int SAMPLES = 50;
				float3 pixelValue = float3(0.0f);
				for (int k = 0; k < SAMPLES; ++k) {
					const Ray ray = generateRay(i, j);
					HitInfo hitInfo;
					if (intersect(hitInfo, ray)) {
						pixelValue += shade(hitInfo, -ray.d);
					} else {
						pixelValue += this->ibl(ray);
					}
				}
				FrameBuffer.pixel(i, j) = pixelValue / SAMPLES;
			}
		}
	}

	void Pathtrace() {
		FrameBuffer.clear();

		// spawn threads
		std::vector<std::thread> threads;
		for (int i = 0; i < numThreads; ++i) {
			threads.push_back(std::thread([this, i]() {
				this->pathtraceSegment(i * globalWidth / numThreads, (i + 1) * globalWidth / numThreads - 1);
			}));
		}

		// join threads
		for (auto& thread : threads) {
			thread.join();
		}
	}

};
static Scene globalScene;

static float fresnel(float eta1, float eta2, float cosThetaI, float cosThetaO) {
	float rhoS = (eta1 * cosThetaI - eta2 * cosThetaO) / (eta1 * cosThetaI + eta2 * cosThetaO);
	float rhoT = (eta1 * cosThetaO - eta2 * cosThetaI) / (eta1 * cosThetaO + eta2 * cosThetaI);
	return (powf(rhoS, 2) + powf(rhoT, 2)) / 2;
}

static float3 reflectRay(const float3& viewDir, const HitInfo& hit, const int level) {
	// steps:
	// 1. find new ray's vector using formula from slides
	// 2. find new ray's first hit using the intersect function
	// 3. call shade on that point
	float3 wi = -viewDir;
	Ray reflectedRay;
	float3 n = hit.N;
	float wn = dot(wi, n);
	if (wn < 0) {
		wn = -wn;
		n = -n;
	}
	reflectedRay.d = -2 * wn * n + wi;
	reflectedRay.o = hit.P - Epsilon * n;
	HitInfo nextHit;
	bool isHit = globalScene.intersect(nextHit, reflectedRay);
	if (isHit) {
		return shade(nextHit, -reflectedRay.d, level+1);
	} else {
		return globalScene.ibl(reflectedRay);
	}
}

static float3 refractRay(const float3& viewDir, const HitInfo& hit, const int level) {
	float3 wi = -viewDir;
	Ray refractedRay;
	float eta1;
	float eta2;
	float wn = dot(wi, hit.N);
	float3 n = hit.N;
	if (wn < 0) {
		eta1 = etaAir;
		eta2 = hit.material->eta;
	} else {
		eta1 = hit.material->eta;
		eta2 = etaAir;
		wn = -wn;
		n = -n;
	}
	float underRoot = 1 - powf(eta1 / eta2, 2) * (1 - powf(wn, 2));

	// Check for total internal reflection
	if (underRoot < 0) {
		return reflectRay(viewDir, hit, level);
	}

	// Refracted ray
	refractedRay.d = (eta1 / eta2) * (wi - wn * n) - (sqrtf(underRoot) * n);
	refractedRay.o = hit.P - Epsilon * n;
	HitInfo nextHit;
	bool isHit = globalScene.intersect(nextHit, refractedRay);
	
	// Fresnel
	float cosThetaI = wn / (length(wi) * length(n));
	float cosThetaO = dot(n, refractedRay.d) / (length(n) * length(refractedRay.d));
	float R = fresnel(eta1, eta2, cosThetaI, cosThetaO);

	if (PCG32::rand() < R) {
		return reflectRay(viewDir, hit, level);
	} else {
		float3 refractedRayValue = (isHit ? shade(nextHit, -refractedRay.d, level+1) : globalScene.ibl(refractedRay));
		return refractedRayValue;
	}

	// return R * reflectRay(viewDir, hit, level) + (1 - R) * refractedRayValue;
}

// generate a cosine-weighted random vector in the hemisphere of w
static float3 cosineWeightedHemisphereSample(const float3& w) {
	// Generate a random point on a unit disk
    float u1 = PCG32::rand();
    float u2 = PCG32::rand();
    float r = sqrt(u1);
    float theta = 2.0f * PI * u2;
    
    float x = r * cos(theta);
    float y = r * sin(theta);

    // Calculate the z coordinate
    float z = sqrt(1.0f - u1);

    // Create the vector
    float3 sample = float3(x, y, z);

    // Transform to the hemisphere defined by the normal
    float3 up = fabs(w.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(up, w));
    float3 bitangent = cross(w, tangent);

    return normalize(tangent * sample.x + bitangent * sample.y + w * sample.z);
}

// generate a uniform random vector in the hemisphere of w
static float3 uniformHemisphereSample(const float3& w) {
	// generate random numbers
	float r1 = 2 * PI * PCG32::rand();
	float r2 = PCG32::rand();
	float r2s = sqrtf(r2);

	// convert to spherical coordinates
	float phi = 2.0 * PI * r1;
	float x = cosf(phi) * r2s;
	float y = sinf(phi) * r2s;
	float z = sqrtf(1 - r2);

	// create coordinate system
	float3 up = float3(0.0f, 0.0f, 1.0f);
	float3 tangent = normalize(cross(up, w));
	float3 bitangent = normalize(cross(w, tangent));

	// convert spherical coordinates to cartesian
	float3 direction = float3(
		tangent[0] * x + bitangent[0] * y + w[0] * z,
		tangent[1] * x + bitangent[1] * y + w[1] * z,
		tangent[2] * x + bitangent[2] * y + w[2] * z
	);

	return normalize(direction);
}

#define cosine_weighted_hemisphere_sampling

static float3 shadeLambertian(const HitInfo& hit, const float3& viewDir, const int level) {
	float3 L = float3(0.0f);
	float3 irradiance;
	
	float3 brdf = hit.material->BRDF();
	if (hit.material->isTextured) {
		brdf *= hit.material->fetchTexture(hit.T);
	}
	// return brdf * PI; // debug output

	float wn = dot(-viewDir, hit.N);
	bool isBackFace = (wn > 0);
	float3 n = (isBackFace ? -hit.N : hit.N);

	// loop over all of the point light sources
	for (int i = 0; i < globalScene.pointLightSources.size(); i++) {
		float3 l = globalScene.pointLightSources[i]->position - hit.P;

		HitInfo dh;
		Ray ray;
		ray.d = -1 * l;
		ray.o = globalScene.pointLightSources[i]->position;
		if (globalScene.intersect(dh, ray) && distance(dh.P, ray.o) + Epsilon * 0.5f < distance(hit.P, ray.o)) {
			continue;
		}

		// the inverse-squared falloff
		const float falloff = length2(l);

		// normalize the light direction
		l /= sqrtf(falloff);

		// get the irradiance
		irradiance = float(std::max(0.0f, dot(n, l)) / (4.0 * PI * falloff)) * globalScene.pointLightSources[i]->wattage;

		L += irradiance * brdf;
	}

	// make a new random ray from here and keep going
	Ray newRay;
#ifdef cosine_weighted_hemisphere_sampling
	newRay.d = cosineWeightedHemisphereSample(n);
#else
	newRay.d = uniformHemisphereSample(n);
#endif
	newRay.o = hit.P + Epsilon * n;
	const float cosTheta = dot(newRay.d, n);

	// probability of the newRay
#ifdef cosine_weighted_hemisphere_sampling
	const float p = cosTheta / PI;
#else
	const float p = 1 / (PI * 2);
#endif

	HitInfo nextHit;
	bool isHit = globalScene.intersect(nextHit, newRay);
	float3 nextColor = (isHit ? shade(nextHit, -newRay.d, level + 1) : globalScene.ibl(newRay));

	return L + nextColor * brdf * cosTheta / p;
}

static float3 shade(const HitInfo& hit, const float3& viewDir, const int level) {
	if (level > maxLevel) {
		return float3(0.0f);
	} else if (hit.material->type == MAT_LAMBERTIAN) {
		return shadeLambertian(hit, viewDir, level);
	} else if (hit.material->type == MAT_METAL) {
		return hit.material->Ks * reflectRay(viewDir, hit, level);
	} else if (hit.material->type == MAT_GLASS) {
		return refractRay(viewDir, hit, level);
	} else {
		// something went wrong - make it apparent that it is an error
		return float3(100.0f, 0.0f, 100.0f);
	}
}

// OpenGL initialization (you will not use any OpenGL/Vulkan/DirectX... APIs to render 3D objects!)
// you probably do not need to modify this in A0 to A3.
class OpenGLInit {
public:
	OpenGLInit() {
		// initialize GLFW
		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW." << std::endl;
			exit(-1);
		}

		// create a window
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
		globalGLFWindow = glfwCreateWindow(globalWidth, globalHeight, "Welcome to CS488/688!", NULL, NULL);
		if (globalGLFWindow == NULL) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// make OpenGL context for the window
		glfwMakeContextCurrent(globalGLFWindow);

		// initialize GLEW
		glewExperimental = true;
		if (glewInit() != GLEW_OK) {
			std::cerr << "Failed to initialize GLEW." << std::endl;
			glfwTerminate();
			exit(-1);
		}

		// set callback functions for events
		glfwSetKeyCallback(globalGLFWindow, keyFunc);
		glfwSetMouseButtonCallback(globalGLFWindow, mouseButtonFunc);
		glfwSetCursorPosCallback(globalGLFWindow, cursorPosFunc);

		// create shader
		FSDraw = glCreateProgram();
		GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(s, 1, &PFSDrawSource, 0);
		glCompileShader(s);
		glAttachShader(FSDraw, s);
		glLinkProgram(FSDraw);

		// create texture
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &GLFrameBufferTexture);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, globalWidth, globalHeight, 0, GL_LUMINANCE, GL_FLOAT, 0);

		// initialize some OpenGL state (will not change)
		glDisable(GL_DEPTH_TEST);

		glUseProgram(FSDraw);
		glUniform1i(glGetUniformLocation(FSDraw, "input_tex"), 0);

		GLint dims[4];
		glGetIntegerv(GL_VIEWPORT, dims);
		const float BufInfo[4] = { float(dims[2]), float(dims[3]), 1.0f / float(dims[2]), 1.0f / float(dims[3]) };
		glUniform4fv(glGetUniformLocation(FSDraw, "BufInfo"), 1, BufInfo);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, GLFrameBufferTexture);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	virtual ~OpenGLInit() {
		glfwTerminate();
	}
};

// main window
// you probably do not need to modify this in A0 to A3.
class CS488Window {
public:
	// put this first to make sure that the glInit's constructor is called before the one for CS488Window
	OpenGLInit glInit;

	CS488Window() {}
	virtual ~CS488Window() {}

	void(*process)() = NULL;

	void start() const {
		if (globalEnableParticles) {
			globalScene.addObject(&globalParticleSystem.particlesMesh);
		}
		globalScene.preCalc();

		// main loop
		while (glfwWindowShouldClose(globalGLFWindow) == GL_FALSE) {
			glfwPollEvents();
			globalViewDir = normalize(globalLookat - globalEye);
			globalRight = normalize(cross(globalViewDir, globalUp));

			if (globalEnableParticles) {
				globalParticleSystem.step();
			}

			// std::clock_t start = std::clock();
			if (globalRenderType == RENDER_RASTERIZE) {
				globalScene.Rasterize();
			} else if (globalRenderType == RENDER_RAYTRACE) {
				globalScene.Raytrace();
			} else if (globalRenderType == RENDER_IMAGE) {
				if (process) process();
			} else if (globalRenderType == RENDER_PATHTRACE) {
				globalScene.Pathtrace();
			}
			// double duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
			// std::cout << duration << std::endl;

			if (globalRecording) {
				unsigned char* buf = new unsigned char[FrameBuffer.width * FrameBuffer.height * 4];
				int k = 0;
				for (int j = FrameBuffer.height - 1; j >= 0; j--) {
					for (int i = 0; i < FrameBuffer.width; i++) {
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).x));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).y));
						buf[k++] = (unsigned char)(255.0f * Image::toneMapping(FrameBuffer.pixel(i, j).z));
						buf[k++] = 255;
					}
				}
				GifWriteFrame(&globalGIFfile, buf, globalWidth, globalHeight, globalGIFdelay);
				delete[] buf;
			}

			// drawing the frame buffer via OpenGL (you don't need to touch this)
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, globalWidth, globalHeight, GL_RGB, GL_FLOAT, &FrameBuffer.pixels[0][0]);
			glRecti(1, 1, -1, -1);
			glfwSwapBuffers(globalGLFWindow);
			globalFrameCount++;
			PCG32::rand();
		}
	}
};

