//
//  Rank.swift
//  RankedTensor
//
//  Copyright 2016-2018 The DLVM Team.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import struct CoreTensor.TensorShape
import struct CoreTensor.TensorSlice

public protocol StaticRank {
    associatedtype UnitType
    associatedtype Shape
    associatedtype ElementTensor
    associatedtype ArrayLiteralElement
    static var rank: UInt { get }
    static func makeTensor(from literal: [ArrayLiteralElement]) -> Tensor<Self>
    static func element(of tensor: Tensor<Self>, at index: Int) -> ElementTensor
    static func element(of tensor: TensorSlice<Self>,
                        at index: Int) -> ElementTensor
    static func updateElement(_ newElement: ElementTensor,
                              at index: Int, in tensor: inout Tensor<Self>)
    static func updateElement(_ newElement: ElementTensor,
                              at index: Int, in tensor: inout TensorSlice<Self>)
}

public typealias Shape1D = (UInt)
public typealias Shape2D = (UInt, UInt)
public typealias Shape3D = (UInt, UInt, UInt)
public typealias Shape4D = (UInt, UInt, UInt, UInt)

extension StaticRank {
    static func staticShape(from shape: TensorShape) -> Shape {
        var elements = shape.map(UInt.init)
        return withUnsafePointer(to: &elements[0]) { ptr in
            ptr.withMemoryRebound(to: Shape.self, capacity: 1) { ptr in
                return ptr.pointee
            }
        }
    }

    static func dynamicShape(from shape: Shape) -> TensorShape {
        var shape = shape
        return withUnsafePointer(to: &shape) { ptr in
            ptr.withMemoryRebound(to: UInt.self, capacity: 1) { ptr in
                let buf = UnsafeBufferPointer(start: ptr, count: Int(rank))
                return TensorShape(buf.lazy.map {Int($0)})
            }
        }
    }
}

public struct Rank1<T> : StaticRank {
    public typealias UnitType = T
    public typealias Shape = Shape1D
    public typealias ElementTensor = T
    public typealias ArrayLiteralElement = T
    public static var rank: UInt { return 1 }

    public static func makeTensor(from literal: [T]) -> Tensor<Rank1<T>> {
        return Tensor<Rank1<T>>(shape: (UInt(literal.count)),
                                units: ContiguousArray(literal))
    }

    public static func makeTensor(with elements: [T]) -> Tensor<Rank1<T>> {
        return makeTensor(from: elements)
    }

    public static func element(of tensor: Tensor<Rank1<T>>,
                               at index: Int) -> T {
        return tensor.units[index]
    }

    public static func element(of tensor: TensorSlice<Rank1<T>>,
                               at index: Int) -> T {
        let unitIndex = index.advanced(by: tensor.units.startIndex)
        return tensor.units[unitIndex]
    }

    public static func updateElement(_ newElement: T, at index: Int,
                                     in tensor: inout Tensor<Rank1<T>>) {
        tensor.base[index] = CoreTensor.TensorSlice(scalar: newElement)
    }

    public static func updateElement(_ newElement: T, at index: Int,
                                     in tensor: inout TensorSlice<Rank1<T>>) {
        tensor.base[index] = CoreTensor.TensorSlice(scalar: newElement)
    }
}

public struct Rank2<T> : StaticRank {
    public typealias UnitType = T
    public typealias Shape = Shape2D
    public typealias ElementTensor = TensorSlice1D<T>
    public typealias ArrayLiteralElement = [T]
    public static var rank: UInt { return 2 }

    public static func makeTensor(from literal: [[T]]) -> Tensor<Rank2<T>> {
        let dim0 = UInt(literal.count)
        guard dim0 > 0 else {
            fatalError("Array literal cannot be empty")
        }
        let dim1 = UInt(literal[0].count)
        guard dim1 > 0 else {
            fatalError("The 2nd dimension cannot be empty")
        }
        let contSize = dim0 * dim1
        var units: ContiguousArray<T> = []
        units.reserveCapacity(Int(contSize))
        for subArray in literal {
            guard subArray.count == dim1 else {
                fatalError("""
                    Element tensors in the 2nd dimension have mismatching shapes
                    """)
            }
            units.append(contentsOf: subArray)
        }
        return Tensor<Rank2<T>>(shape: (dim0, dim1), units: units)
    }

    public static func makeTensor(
        with elements: [Tensor<Rank1<T>>]) -> Tensor<Rank2<T>> {
        return makeTensor(from: elements.map { Array($0.units) })
    }

    public static func makeTensor(
        with elements: [TensorSlice<Rank1<T>>]) -> Tensor<Rank2<T>> {
        return makeTensor(from: elements.map { Array($0.units) })
    }

    public static func element(
        of tensor: Tensor<Rank2<T>>, at index: Int) -> TensorSlice<Rank1<T>> {
        return TensorSlice1D(base: tensor, index: index)
    }

    public static func element(
        of tensor: TensorSlice<Rank2<T>>,
        at index: Int
    ) -> TensorSlice<Rank1<T>> {
        return TensorSlice1D(base: tensor, index: index)
    }

    public static func updateElement(_ newElement: TensorSlice<Rank1<T>>,
                                     at index: Int,
                                     in tensor: inout Tensor<Rank2<T>>) {
        tensor.base[index] = newElement.base
    }

    public static func updateElement(_ newElement: TensorSlice<Rank1<T>>,
                                     at index: Int,
                                     in tensor: inout TensorSlice<Rank2<T>>) {
        tensor.base[index] = newElement.base
    }
}

public struct Rank3<T> : StaticRank {
    public typealias UnitType = T
    public typealias Shape = Shape3D
    public typealias ElementTensor = TensorSlice2D<T>
    public typealias ArrayLiteralElement = [[T]]
    public static var rank: UInt { return 3 }

    public static func makeTensor(from literal: [[[T]]]) -> Tensor<Rank3<T>> {
        let dim0 = UInt(literal.count)
        guard dim0 > 0 else {
            fatalError("Array literal cannot be empty")
        }
        let dim1 = UInt(literal[0].count)
        guard dim1 > 0 else {
            fatalError("The 2nd dimension cannot be empty")
        }
        let dim2 = UInt(literal[0][0].count)
        guard dim2 > 0 else {
            fatalError("The 3rd dimension cannot be empty")
        }
        let contSize = dim0 * dim1 * dim2
        var units: ContiguousArray<T> = []
        units.reserveCapacity(Int(contSize))
        for subArray in literal {
            guard subArray.count == dim1 else {
                fatalError("""
                    Element tensors in the 2nd dimension have mismatching shapes
                    """)
            }
            for subSubArray in subArray {
                guard subSubArray.count == dim2 else {
                    fatalError("""
                        Element tensors in the 3nd dimension have mismatching \
                        shapes
                    """)
                }
                units.append(contentsOf: subSubArray)
            }
        }
        return Tensor<Rank3<T>>(shape: (dim0, dim1, dim2), units: units)
    }

    public static func makeTensor(
        with elements: [Tensor<Rank2<T>>]) -> Tensor<Rank3<T>> {
        return makeTensor(from: elements.map { $0.map { Array($0.units) } })
    }

    public static func makeTensor(
        with elements: [TensorSlice<Rank2<T>>]) -> Tensor<Rank3<T>> {
        return makeTensor(from: elements.map { $0.map { Array($0.units) } })
    }

    public static func element(
        of tensor: Tensor<Rank3<T>>, at index: Int) -> TensorSlice<Rank2<T>> {
        return TensorSlice2D(base: tensor, index: index)
    }

    public static func element(of tensor: TensorSlice<Rank3<T>>,
                               at index: Int) -> TensorSlice<Rank2<T>> {
        return TensorSlice2D(base: tensor, index: index)
    }

    public static func updateElement(
        _ newElement: TensorSlice<Rank2<T>>,
        at index: Int, in tensor: inout Tensor<Rank3<T>>
    ) {
        tensor.base[index] = newElement.base
    }

    public static func updateElement(
        _ newElement: TensorSlice<Rank2<T>>,
        at index: Int, in tensor: inout TensorSlice<Rank3<T>>
    ) {
        tensor.base[index] = newElement.base
    }
}

public struct Rank4<T> : StaticRank {
    public typealias UnitType = T
    public typealias Shape = Shape4D
    public typealias ElementTensor = TensorSlice3D<T>
    public typealias ArrayLiteralElement = [[[T]]]
    public static var rank: UInt { return 4 }

    public static func makeTensor(from literal: [[[[T]]]]) -> Tensor<Rank4<T>> {
        let dim0 = UInt(literal.count)
        guard dim0 > 0 else {
            fatalError("Array literal cannot be empty")
        }
        let dim1 = UInt(literal[0].count)
        guard dim1 > 0 else {
            fatalError("The 2nd dimension cannot be empty")
        }
        let dim2 = UInt(literal[0][0].count)
        guard dim2 > 0 else {
            fatalError("The 3rd dimension cannot be empty")
        }
        let dim3 = UInt(literal[0][0][0].count)
        guard dim3 > 0 else {
            fatalError("The 3rd dimension cannot be empty")
        }
        let contSize = dim0 * dim1 * dim2 * dim3
        var units: ContiguousArray<T> = []
        units.reserveCapacity(Int(contSize))
        for subArray in literal {
            guard subArray.count == dim1 else {
                fatalError("""
                    Element tensors in the 2nd dimension have mismatching shapes
                    """)
            }
            for subSubArray in subArray {
                guard subSubArray.count == dim2 else {
                    fatalError("""
                        Element tensors in the 3nd dimension have mismatching \
                        shapes
                        """)
                }
                for subSubSubArray in subSubArray {
                    guard subSubSubArray.count == dim3 else {
                        fatalError("""
                            Element tensors in the 4nd dimension have \
                            mismatching shapes
                            """)
                    }
                    units.append(contentsOf: subSubSubArray)
                }
            }
        }
        return Tensor<Rank4<T>>(shape: (dim0, dim1, dim2, dim3), units: units)
    }

    public static func makeTensor(
        with elements: [Tensor<Rank3<T>>]) -> Tensor<Rank4<T>> {
        return makeTensor(from: elements.map {
            $0.map { $0.map { Array($0.units) } }
        })
    }

    public static func makeTensor(
        with elements: [TensorSlice<Rank3<T>>]) -> Tensor<Rank4<T>> {
        return makeTensor(from: elements.map {
            $0.map { $0.map { Array($0.units) } }
        })
    }

    public static func element(of tensor: Tensor<Rank4<T>>,
                               at index: Int) -> TensorSlice<Rank3<T>> {
        return TensorSlice3D(base: tensor, index: index)
    }

    public static func element(of tensor: TensorSlice<Rank4<T>>,
                               at index: Int) -> TensorSlice<Rank3<T>> {
        return TensorSlice3D(base: tensor, index: index)
    }

    public static func updateElement(
        _ newElement: TensorSlice<Rank3<T>>,
        at index: Int, in tensor: inout Tensor<Rank4<T>>
    ) {
        tensor.base[index] = newElement.base
    }

    public static func updateElement(
        _ newElement: TensorSlice<Rank3<T>>,
        at index: Int, in tensor: inout TensorSlice<Rank4<T>>
    ) {
        tensor.base[index] = newElement.base
    }
}
