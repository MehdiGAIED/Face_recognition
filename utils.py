


def graph_PB_vis(pb_path= 'model/20170511-185253.pb'):
    """ 
    Return graph layers for .pb weights file 
    to visualise the layers run under main __name__ 
    print(''.join(name +'\n' for name in names))
    """
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow.python.platform import gfile
    GRAPH_PB_PATH = pb_path
    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes=[n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
    
    return names

if __name__=="__main__":
    names=graph_PB_vis()
    print(''.join(name +'\n' for name in names))



