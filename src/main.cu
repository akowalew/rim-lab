#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <GL/glut.h>

constexpr auto cr = -0.123;
constexpr auto ci = 0.745;
const auto abs_c = std::hypot(ci, cr);
const auto R_2 = (abs_c > 2) ? (abs_c * abs_c) : 4;

constexpr auto DIM = 1000; /* rozmiar rysunku w pikselach */

int julia(float x, float y)
{
    for(auto i = 0; i < DIM; ++i)
    {
        auto x_2 = x * x;
        auto y_2 = y * y;
        if(x_2 + y_2 > R_2)
        {
            // Uciekinierzy
            return 0;
        }

        y = 2 * x * y + ci;
        x = x_2 - y_2 + cr;
    }

    // Więżniowie
    return 1;
}

void kernel(unsigned char *ptr,
    const int xw, const int yw,
    const float dx, const float dy,
    const float scale)
{
    /* przeliczenie współrzędnych pikselowych (xw, yw)
    na matematyczne (x, y) z uwzględnieniem skali
    (scale) i matematycznego środka (dx, dy) */
    auto x = scale * (xw - DIM/2) / (DIM/2) + dx,
    y = scale * (yw - DIM/2) / (DIM/2) + dy;
    auto offset /* w buforze pikseli */ = xw + yw*DIM;

    /* kolor: czarny dla uciekinierów (julia == 0)
    czerwony dla więźniów (julia == 1) */
    ptr[offset*4 + 0 /* R */] = (unsigned char) (255*julia(x,y));
    ptr[offset*4 + 1 /* G */] = 0;
    ptr[offset*4 + 2 /* B */] = 0;
    ptr[offset*4 + 3 /* A */] = 255;
}
/**************************************************/

static unsigned char pixbuf[DIM * DIM * 4];
static float dx = 0.0f, dy = 0.0f;
static float scale = 1.5f;

static void disp(void)
{
    glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, pixbuf);
    glutSwapBuffers();
}

static void recompute(void)
{
    for (auto yw = 0; yw < DIM; yw++)
    {
        for (auto xw = 0; xw < DIM; xw++)
        {
            kernel(pixbuf, xw, yw, dx, dy, scale);
        }
    }

    glutPostRedisplay();
}

static void kbd(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'p':
            dx += scale * (x - DIM/2) / (DIM/2);
            dy -= scale * (y - DIM/2) / (DIM/2);
            break;
        case 'z':
            scale *= 0.80f;
            break;
        case 'Z':
            scale *= 1.25f;
            break;
        case '=':
            scale = 1.50f;
            dx = dy = 0.0f;
            break;
        case 27:
            /* Esc */ exit(0);
    }

    recompute();
}

int main(int argc, char *argv[])
{
    checkCudaErrors(cudaSetDevice(0));

    glutInit(&argc, argv); /* inicjacja biblioteki GLUT */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); /* opcje */
    glutInitWindowSize(DIM, DIM); /* rozmiar okna graficznego */
    glutCreateWindow("RIM - fraktal Julii"); /* tytuł okna */
    glutDisplayFunc(disp); /* funkcja zwrotna zobrazowania */
    glutKeyboardFunc(kbd); /* funkcja zwrotna klawiatury */
    recompute(); /* obliczenie pierwszego rysunku */
    glutMainLoop(); /* główna pętla obsługi zdarzeń */
}
