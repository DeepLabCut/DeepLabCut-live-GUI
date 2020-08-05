import multiprocess as mp
from multiprocess import queues
from queue import Queue, Empty, Full


class QueuePositionError(Exception):
    """ Error in position argument of queue read """

    pass


class ClearableQueue(Queue):
    """ A Queue that provides safe methods for writing to a full queue, reading to an empty queue, and a method to clear the queue """

    def __init__(self, maxsize=0):

        super().__init__(maxsize)

    def clear(self):
        """ Clears queue, returns all objects in a list

        Returns
        -------
        list
            list of objects from the queue
        """

        objs = []

        try:
            while True:
                objs.append(self.get_nowait())
        except Empty:
            pass

        return objs

    def write(self, obj, clear=False):
        """ Puts an object in the queue, with an option to clear queue before writing.

        Parameters
        ----------     
        obj : [type]
            An object to put in the queue
        clear : bool, optional
            flag to clear queue before putting, by default False
        
        Returns
        -------
        bool
            if write was sucessful, returns True
        """

        if clear:
            self.clear()

        try:
            self.put_nowait(obj)
            success = True
        except Full:
            success = False

        return success

    def read(self, clear=False, position="last"):
        """ Gets an object in the queue, with the option to clear the queue and return the first element, last element, or all elements
        
        Parameters
        ----------
        clear : bool, optional
            flag to clear queue before putting, by default False
        position : str, optional
            If clear is True, returned object depends on position. 
            If position = "last", returns last object. 
            If position = "first", returns first object. 
            If position = "all", returns all objects from the queue.
        
        Returns
        -------
        object
            object retrieved from the queue
        """

        obj = None

        if clear:

            objs = self.clear()

            if len(objs) > 0:
                if position == "first":
                    obj = objs[0]
                elif position == "last":
                    obj = objs[-1]
                elif position == "all":
                    obj = objs
                else:
                    raise QueuePositionError(
                        "Queue read position should be one of 'first', 'last', or 'all'"
                    )
        else:

            try:
                obj = self.get_nowait()
            except Empty:
                pass

        return obj


class ClearableMPQueue(mp.queues.Queue):
    """ A multiprocess Queue that provides safe methods for writing to a full queue, reading to an empty queue, and a method to clear the queue """

    def __init__(self, maxsize=0, ctx=mp.get_context("spawn")):

        super().__init__(maxsize, ctx=ctx)

    def clear(self):
        """ Clears queue, returns all objects in a list

        Returns
        -------
        list
            list of objects from the queue
        """

        objs = []

        try:
            while True:
                objs.append(self.get_nowait())
        except Empty:
            pass

        return objs

    def write(self, obj, clear=False):
        """ Puts an object in the queue, with an option to clear queue before writing.

        Parameters
        ----------     
        obj : [type]
            An object to put in the queue
        clear : bool, optional
            flag to clear queue before putting, by default False
        
        Returns
        -------
        bool
            if write was sucessful, returns True
        """

        if clear:
            self.clear()

        try:
            self.put_nowait(obj)
            success = True
        except Full:
            success = False

        return success

    def read(self, clear=False, position="last"):
        """ Gets an object in the queue, with the option to clear the queue and return the first element, last element, or all elements
        
        Parameters
        ----------
        clear : bool, optional
            flag to clear queue before putting, by default False
        position : str, optional
            If clear is True, returned object depends on position. 
            If position = "last", returns last object. 
            If position = "first", returns first object. 
            If position = "all", returns all objects from the queue.
        
        Returns
        -------
        object
            object retrieved from the queue
        """

        obj = None

        if clear:

            objs = self.clear()

            if len(objs) > 0:
                if position == "first":
                    obj = objs[0]
                elif position == "last":
                    obj = objs[-1]
                elif position == "all":
                    obj = objs
                else:
                    raise QueuePositionError(
                        "Queue read position should be one of 'first', 'last', or 'all'"
                    )

        else:

            try:
                obj = self.get_nowait()
            except Empty:
                pass

        return obj
