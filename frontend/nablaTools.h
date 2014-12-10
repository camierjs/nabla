// NABLA - a Numerical Analysis Based LAnguage

// Copyright (C) 2014 CEA/DAM/DIF
// Jean-Sylvain CAMIER - Jean-Sylvain.Camier@cea.fr

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// See the LICENSE file for details.
#ifndef _NABLA_TOOLS_H_
#define _NABLA_TOOLS_H_

int nprintf(const struct nablaMainStruct *nabla, const char *debug, const char *format, ...);

int hprintf(const struct nablaMainStruct *nabla, const char *debug, const char *format, ...);

char *toolStrDownCase(const char *);

char *toolStrUpCase(const char *);

char *trQuote(const char * str);

char *op2name(char *op);

void nUtf8(char**);

int nablaMakeTempFile(const char *, char **);

void nUtf8SupThree(char **);

#endif // _NABLA_TOOLS_H_
