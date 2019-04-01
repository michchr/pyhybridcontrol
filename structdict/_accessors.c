/* ------------------------------------------------------------------------- */

#include "Python.h"
#include "structmember.h"

/* ------------------------------------------------------------------------- */


/* We link this module statically for convenience.  If compiled as a shared
   library instead, some compilers don't allow addresses of Python objects
   defined in other libraries to be used in static initializers here.  The
   DEFERRED_ADDRESS macro is used to tag the slots where such addresses
   appear; the module init function must fill in the tagged slots at runtime.
   The argument is for documentation -- the macro ignores it.
*/
#define DEFERRED_ADDRESS(ADDR) 0


/*
 * BEGIN AttributeAccessor type definition
 */


typedef struct {
    PyObject_HEAD
    PyObject *attr_accessor_name;
    PyObject *attr_accessor_doc;
} AttributeAccessorObject;


static PyMemberDef attr_accessor_members[] = {
        {"name",    T_OBJECT, offsetof(AttributeAccessorObject, attr_accessor_name), READONLY},
        {"__doc__", T_OBJECT, offsetof(AttributeAccessorObject, attr_accessor_doc), 0},
        {0}
};


static void
attr_accessor_dealloc(PyObject *self) {
    AttributeAccessorObject *gs = (AttributeAccessorObject *) self;

    _PyObject_GC_UNTRACK(self);
    Py_XDECREF(gs->attr_accessor_name);
    Py_XDECREF(gs->attr_accessor_doc);
    self->ob_type->tp_free(self);
}


static PyObject *
attr_accessor_get(PyObject *self, PyObject *obj, PyObject *type) {
    PyObject *ret = NULL;
    AttributeAccessorObject *gs = (AttributeAccessorObject *) self;
    PyObject *dict;

    if (obj == NULL || obj == Py_None) {
        Py_INCREF(self);
        return self;
    }
    if (gs->attr_accessor_name == NULL) {
        PyErr_SetString(PyExc_AttributeError, "unreadable attribute");
        return NULL;
    }

    Py_XINCREF(gs->attr_accessor_name);
    Py_INCREF(obj);
    dict = PyObject_GenericGetDict(obj, NULL);


    if (dict != NULL) {
        if (PyDict_Size(dict) > 0) {
            ret = PyDict_GetItem(dict, gs->attr_accessor_name);
            if (ret != NULL) {
                Py_INCREF(ret);
                goto done;
            }
        }

        PyErr_Format(PyExc_AttributeError,
                     "'%.100s' object has no attribute '%U' in instance '__dict__'",
                     Py_TYPE(obj)->tp_name, gs->attr_accessor_name);

    } else if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        PyErr_Format(PyExc_AttributeError,
                     "'%.100s' object has no instance '__dict__'",
                     Py_TYPE(obj)->tp_name);
    }

    done:
    Py_XDECREF(dict);
    Py_XDECREF(gs->attr_accessor_name);
    Py_DECREF(obj);
    return ret;
}


static int
attr_accessor_set(PyObject *self, PyObject *obj, PyObject *value) {
    int ret = -1;
    AttributeAccessorObject *gs = (AttributeAccessorObject *) self;
    PyObject *dict;

    Py_XINCREF(gs->attr_accessor_name);
    Py_INCREF(obj);
    dict = PyObject_GenericGetDict(obj, NULL);

    if (dict != NULL) {
        if (value != NULL) {
            ret = PyDict_SetItem(dict, gs->attr_accessor_name, value);
        } else {
            ret = PyDict_DelItem(dict, gs->attr_accessor_name);
        }
        if (!ret) {
            goto done;
        } else {
            PyErr_Format(PyExc_AttributeError,
                         "'%.100s' object has no attribute '%U' in instance '__dict__'",
                         Py_TYPE(obj)->tp_name, gs->attr_accessor_name);
        }
    } else if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
        PyErr_Format(PyExc_AttributeError,
                     "'%.100s' object has no instance '__dict__'",
                     Py_TYPE(obj)->tp_name);
    }


    done:
    Py_XDECREF(dict);
    Py_XDECREF(gs->attr_accessor_name);
    Py_DECREF(obj);
    return ret;
}


static int
attr_accessor_init(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *name = NULL, *doc = NULL;
    static char *kwlist[] = {"name", "doc", 0};
    AttributeAccessorObject *gs = (AttributeAccessorObject *) self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O:AttributeAccessor",
                                     kwlist, &name, &doc))
        return -1;

    if (!PyUnicode_Check(name)) {
        PyErr_Format(PyExc_TypeError,
                     "'name' must be of type string or a subclass, not: '%.200s'",
                     name->ob_type->tp_name);
        return -1;
    }

    if (doc == NULL || doc == Py_None) {
        Py_INCREF(name);
        doc = PyUnicode_FromFormat("Attribute accessor for variable %R", name);
    } else if (!PyUnicode_Check(doc)) {
        PyErr_Format(PyExc_TypeError,
                     "'doc' must be of type string or a subclass, not: '%.200s'",
                     doc->ob_type->tp_name);
        return -1;
    } else {
        Py_INCREF(name);
        Py_INCREF(doc);
    }


    Py_XSETREF(gs->attr_accessor_name, name);
    Py_XSETREF(gs->attr_accessor_doc, doc);

    return 0;
}


PyDoc_STRVAR(set_name_doc,
             "Method to verify that self.name is matches identifier");

static PyObject *
attr_accessor_set_name(PyObject *self, PyObject *args) {
    PyObject *type, *name;
    AttributeAccessorObject *gs = (AttributeAccessorObject *) self;

    if (!PyArg_ParseTuple(args, "OO:__set_name__", &type, &name)) {
        return NULL;
    }
    Py_INCREF(type);
    Py_INCREF(name);

    if (!PyObject_RichCompareBool(name, gs->attr_accessor_name, Py_EQ)) {
        PyErr_Format(PyExc_ValueError,
                     "name:%R of '%.100s' object in class '%.100s' does not match self.name:%R",
                     gs->attr_accessor_name, Py_TYPE(gs)->tp_name, ((PyTypeObject *) type)->tp_name,
                     gs->attr_accessor_name);
        Py_DECREF(type);
        Py_DECREF(name);
        return NULL;
    }
    Py_XDECREF(type);
    Py_XDECREF(name);
    Py_RETURN_NONE;
}

static PyMethodDef attr_accessor_methods[] = {
        {"__set_name__", attr_accessor_set_name, METH_VARARGS, set_name_doc},
        {0}
};


PyDoc_STRVAR(attr_accessor_doc,
             "AttributeAccessor(name: str, doc=None) -> AttributeAccessor attribute\n"
);

static int
attr_accessor_traverse(PyObject *self, visitproc visit, void *arg) {
    AttributeAccessorObject *aa = (AttributeAccessorObject *) self;
    Py_VISIT(aa->attr_accessor_name);
    Py_VISIT(aa->attr_accessor_doc);
    return 0;
}

static int
attr_accessor_clear(PyObject *self) {
    AttributeAccessorObject *aa = (AttributeAccessorObject *) self;
    Py_CLEAR(aa->attr_accessor_name);
    Py_CLEAR(aa->attr_accessor_doc);
    return 0;
}

PyTypeObject AttributeAccessor_Type = {
        PyVarObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type), 0)
        "accessors.AttributeAccessor",              /* tp_name */
        sizeof(AttributeAccessorObject),            /* tp_basicsize */
        0,                                          /* tp_itemsize */
        /* methods */
        (destructor) attr_accessor_dealloc,         /* tp_dealloc */
        0,                                          /* tp_print */
        0,                                          /* tp_getattr */
        0,                                          /* tp_setattr */
        0,                                          /* tp_reserved */
        0,                                          /* tp_repr */
        0,                                          /* tp_as_number */
        0,                                          /* tp_as_sequence */
        0,                                          /* tp_as_mapping */
        0,                                          /* tp_hash */
        0,                                          /* tp_call */
        0,                                          /* tp_str */
        (getattrofunc) PyObject_GenericGetAttr,     /* tp_getattro */
        0,                                          /* tp_setattro */
        0,                                          /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
            Py_TPFLAGS_BASETYPE,                    /* tp_flags */
        attr_accessor_doc,                          /* tp_doc */
        (traverseproc) attr_accessor_traverse,      /* tp_traverse */
        (inquiry) attr_accessor_clear,              /* tp_clear */
        0,                                          /* tp_richcompare */
        0,                                          /* tp_weaklistoffset */
        0,                                          /* tp_iter */
        0,                                          /* tp_iternext */
        attr_accessor_methods,                      /* tp_methods */
        attr_accessor_members,                      /* tp_members */
        0,                                          /* tp_getset */
        0,                                          /* tp_base */
        0,                                          /* tp_dict */
        (descrgetfunc) attr_accessor_get,           /* tp_descr_get */
        (descrsetfunc) attr_accessor_set,           /* tp_descr_set */
        0,                                          /* tp_dictoffset */
        (initproc) attr_accessor_init,              /* tp_init */
        (allocfunc) PyType_GenericAlloc,            /* tp_alloc */
        (newfunc) PyType_GenericNew,                /* tp_new */
        (freefunc) PyObject_GC_Del,                 /* tp_free */
};



/*
 *END AttributeAccessor type definition
 */


/*
 * BEGIN ItemAccessorMixin type definition
 */

typedef struct {
    PyObject_HEAD
} ItemAccessorMixinObject;

PyTypeObject ItemAccessorMixin_Type;

static PyObject *
item_accessor_get_attro(PyObject *obj, PyObject *name) {
    PyTypeObject *tp = Py_TYPE(obj);
    PyMappingMethods *m;
    PyObject *descr = NULL;
    PyObject *res = NULL;
    descrgetfunc f;
    Py_ssize_t dictoffset;
    PyObject *dict = NULL;
    PyObject **dictptr;

    if (!PyUnicode_Check(name)) {
        PyErr_Format(PyExc_TypeError,
                     "attribute name must be string, not '%.200s'",
                     name->ob_type->tp_name);
        return NULL;
    }
    Py_INCREF(name);

    if (tp->tp_dict == NULL) {
        if (PyType_Ready(tp) < 0)
            goto done;
    }
    m = tp->tp_as_mapping;
    descr = _PyType_Lookup(tp, name);

    f = NULL;
    if (descr != NULL) {  /*Check for class attribute*/
        Py_INCREF(descr);
        f = descr->ob_type->tp_descr_get;
        if (f != NULL && PyDescr_IsData(descr)) {
            res = f(descr, obj, (PyObject *) obj->ob_type);
            goto done;
        }
    } else if (m && m->mp_subscript) { /*if mapping and no class attribute override lookup*/
        if ((res = m->mp_subscript(obj, name))!=NULL) {
            goto done;
        } else if (PyErr_ExceptionMatches(PyExc_KeyError)) {
            PyErr_Clear();
            if (strcmp(PyUnicode_AsUTF8(name), "__dict__") != 0) {
                PyErr_Format(PyExc_AttributeError,
                             "'%.100s' object has no item with key '%U'.\n"
                             "Note: Item may still be present in instance '__dict__' if it exists.",
                             tp->tp_name, name);
                goto done;
            }
            goto attr_err;
        } else {
            goto done;
        }
    }

    /* Continue standard attribute lookup if not a mapping*/
    if (m == NULL) {
        /* Inline _PyObject_GetDictPtr */
        dictoffset = tp->tp_dictoffset;
        if (dictoffset != 0) {
            if (dictoffset < 0) {
                Py_ssize_t tsize;
                size_t size;

                tsize = ((PyVarObject *) obj)->ob_size;
                if (tsize < 0)
                    tsize = -tsize;
                size = _PyObject_VAR_SIZE(tp, tsize);
                assert(size <= PY_SSIZE_T_MAX);

                dictoffset += (Py_ssize_t) size;
                assert(dictoffset > 0);
                assert(dictoffset % SIZEOF_VOID_P == 0);
            }
            dictptr = (PyObject **) ((char *) obj + dictoffset);
            dict = *dictptr;
        }
    }

    if (dict != NULL) {
        Py_INCREF(dict);
        res = PyDict_GetItem(dict, name);
        if (res != NULL) {
            Py_INCREF(res);
            Py_DECREF(dict);
            goto done;
        }
        Py_DECREF(dict);
    }

    if (f != NULL) {
        res = f(descr, obj, (PyObject *) Py_TYPE(obj));
        goto done;
    }

    if (descr != NULL) {
        res = descr;
        descr = NULL;
        goto done;
    }

    attr_err:
    PyErr_Format(PyExc_AttributeError,
                 "'%.50s' object has no attribute '%U'",
                 tp->tp_name, name);
    done:
    Py_XDECREF(descr);
    Py_DECREF(name);
    return res;
}


static int
item_accessor_set_attro(PyObject *obj, PyObject *name, PyObject *value) {
    PyTypeObject *tp = Py_TYPE(obj);
    PyMappingMethods *m;
    PyObject *descr;
    descrsetfunc f;
    PyObject *dict = NULL;
    int res = -1;

    if (!PyUnicode_Check(name)) {
        PyErr_Format(PyExc_TypeError,
                     "attribute name must be string, not '%.200s'",
                     name->ob_type->tp_name);
        return -1;
    }

    if (tp->tp_dict == NULL && PyType_Ready(tp) < 0)
        return -1;

    Py_INCREF(name);
    Py_INCREF(obj);

    m = tp->tp_as_mapping;
    descr = _PyType_Lookup(tp, name);

    if (descr != NULL) { /*Check for class attribute*/
        Py_INCREF(descr);
        f = descr->ob_type->tp_descr_set;
        if (f != NULL) {
            res = f(descr, obj, value);
            goto done;
        }
    } else if (m && m->mp_ass_subscript) { /*if mapping and no class attribute override set_item*/
        if ((res = m->mp_ass_subscript(obj, name, value)) == 0) {
            goto done;
        } else if (res < 0 && strcmp(PyUnicode_AsUTF8(name), "__dict__") != 0) {
            goto done;
        }
    }

    if (m == NULL) {
        dict = PyObject_GenericGetDict(obj, NULL);
        if (dict == NULL  && !PyErr_ExceptionMatches(PyExc_AttributeError)) {
            goto done;
        }
    }

    if (dict == NULL) {
        if (descr == NULL) {
            PyErr_Format(PyExc_AttributeError,
                         "'%.100s' object has no attribute '%U'",
                         tp->tp_name, name);
        } else {
            PyErr_Format(PyExc_AttributeError,
                         "'%.50s' object attribute '%U' is read-only",
                         tp->tp_name, name);
        }
        goto done;
    }
    res = PyDict_SetItem(dict, name, value);
    Py_XDECREF(dict);

    if (res < 0 && PyErr_ExceptionMatches(PyExc_KeyError)) {
        PyErr_Format(PyExc_AttributeError,
                     "'%.100s' object has no attribute '%U'",
                     tp->tp_name, name);
    }

    done:
    Py_XDECREF(descr);
    Py_DECREF(obj);
    Py_DECREF(name);
    return res;
}



PyDoc_STRVAR(item_accessor_doc,
             "ItemAccessorMixin() -> ItemAccessorMixin object\n"
);

PyTypeObject ItemAccessorMixin_Type = {
        PyVarObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type), 0)
        "accessors.ItemAccessorMixin",                  /* tp_name */
        sizeof(ItemAccessorMixinObject),                /* tp_basicsize */
        0,                                              /* tp_itemsize */
        /* methods */
        0,                                              /* tp_dealloc */
        0,                                              /* tp_print */
        0,                                              /* tp_getattr */
        0,                                              /* tp_setattr */
        0,                                              /* tp_reserved */
        0,                                              /* tp_repr */
        0,                                              /* tp_as_number */
        0,                                              /* tp_as_sequence */
        0,                                              /* tp_as_mapping */
        0,                                              /* tp_hash */
        0,                                              /* tp_call */
        0,                                              /* tp_str */
        (getattrofunc) item_accessor_get_attro,         /* tp_getattro */
        (setattrofunc) item_accessor_set_attro,         /* tp_setattro */
        0,                                              /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,       /* tp_flags */
        item_accessor_doc,                              /* tp_doc */
        0,                                              /* tp_traverse */
        0,                                              /* tp_clear */
        0,                                              /* tp_richcompare */
        0,                                              /* tp_weaklistoffset */
        0,                                              /* tp_iter */
        0,                                              /* tp_iternext */
        0,                                              /* tp_methods */
        0,                                              /* tp_members */
        0,                                              /* tp_getset */
        0,                                              /* tp_base */
        0,                                              /* tp_dict */
        0,                                              /* tp_descr_get */
        0,                                              /* tp_descr_set */
        0,                                              /* tp_dictoffset */
        0,                                              /* tp_init */
        (allocfunc) PyType_GenericAlloc,                /* tp_alloc */
        0,     /*inherited from base object*/           /* tp_new */
        (freefunc) PyObject_Del,                        /* tp_free */
};


/*
 *END ItemAccessorMixin type definition
 */



static int
accessor_type_exec(PyObject *m) {
    /* Fill in deferred data addresses.  This must be done before
       PyType_Ready() is called.  Note that PyType_Ready() automatically
       initializes the ob.ob_type field to &PyType_Type if it's NULL,
       so it's not necessary to fill in ob_type first. */

    AttributeAccessor_Type.tp_base = &PyBaseObject_Type;
    if (PyType_Ready(&AttributeAccessor_Type) < 0)
        goto fail;

    Py_INCREF(&AttributeAccessor_Type);
    if (PyModule_AddObject(m, "AttributeAccessor",
                           (PyObject *) &AttributeAccessor_Type) < 0)
        goto fail;

    ItemAccessorMixin_Type.tp_base = &PyBaseObject_Type;
    if (PyType_Ready(&ItemAccessorMixin_Type) < 0)
        goto fail;

    Py_INCREF(&ItemAccessorMixin_Type);
    if (PyModule_AddObject(m, "ItemAccessorMixin", (PyObject *) &ItemAccessorMixin_Type) < 0)
        goto fail;

    return 0;

    fail:
    Py_XDECREF(&AttributeAccessor_Type);
    Py_XDECREF(&ItemAccessorMixin_Type);
    Py_XDECREF(m);
    return -1;
}

static struct PyModuleDef_Slot accessors_type_slots[] = {
        {Py_mod_exec, accessor_type_exec},
        {0, NULL},
};


static struct PyModuleDef _accessors_module = {
        PyModuleDef_HEAD_INIT,                          /* m_base */
        "_accessors",                                   /* m_name */
        0,                                              /* m_doc */
        0,                                              /* m_size */
        0,                                              /* m_methods */
        (PyModuleDef_Slot *) accessors_type_slots,      /* m_slots */
        NULL,                                           /* m_traverse */
        NULL,                                           /* m_clear */
        NULL                                            /* m_free */
};



PyMODINIT_FUNC
PyInit__accessors(void) {
    return PyModuleDef_Init(&_accessors_module);
}