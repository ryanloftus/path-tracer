#include "cs488.h"
CS488Window CS488;

// loading .obj file from the command line arguments
static TriangleMesh mesh;
static void setupScene(int argc, const char* argv[]) {
    if (argc > 1) {
        bool objLoadSucceed = mesh.load(argv[1]);
        if (!objLoadSucceed) {
            printf("Invalid .obj file.\n");
            printf("Making a single triangle instead.\n");
            mesh.createSingleTriangle();
        }
    } else {
        printf("Specify .obj file in the command line arguments. Example: CS488.exe cornellbox.obj\n");
        printf("Making a single triangle instead.\n");
        mesh.createSingleTriangle();
    }
    if (argc > 2) {
        try {
            globalScene.envImage.load(argv[2]);
        } catch (...) {
            printf("Invalid hdr file.");
        }
    }
    globalScene.addObject(&mesh);
}

static void Project(int argc, const char* argv[]) {
    setupScene(argc, argv);
}


int main(int argc, const char* argv[]) {
    Project(argc, argv);
    CS488.start();
}
