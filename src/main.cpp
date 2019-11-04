#include <GL/glut.h>

#include "fractals.cuh"

static unsigned char pixbuf[DIM * DIM * 4];
static float dx = 0.0f, dy = 0.0f;
static float scale = 1.5f;

static void disp()
{
    glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, pixbuf);
    glutSwapBuffers();
}

static void recompute()
{
    fractals::compute_julia(pixbuf, dx, dy, scale);
    glutPostRedisplay();
}

static void kbd(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'p':
            dx += scale * (x - DIM/2) / (DIM/2);
            break;
        case 'o':
            dx -= scale * (x - DIM/2) / (DIM/2);
            break;
        case 'i':
            dy -= scale * (y - DIM/2) / (DIM/2);
            break;
        case 'u':
            dy += scale * (y - DIM/2) / (DIM/2);
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
    fractals::init();

    glutInit(&argc, argv); /* inicjacja biblioteki GLUT */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); /* opcje */
    glutInitWindowSize(DIM, DIM); /* rozmiar okna graficznego */
    glutCreateWindow("RIM - fraktal Julii"); /* tytuł okna */
    glutDisplayFunc(disp); /* funkcja zwrotna zobrazowania */
    glutKeyboardFunc(kbd); /* funkcja zwrotna klawiatury */

    recompute(); /* obliczenie pierwszego rysunku */
    glutMainLoop(); /* główna pętla obsługi zdarzeń */

    fractals::cleanup();
}
