//
//  Tensor.swift
//  Tensor
//
//  Copyright 2016-2017 The DLVM Team.
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
import struct CoreTensor.Tensor

/// Tensor with static rank
public struct Tensor<Rank: StaticRank> : RankedTensorProtocol {
    public typealias UnitType = Rank.UnitType
    public typealias Shape = Rank.Shape
    public typealias BaseForm = Tensor<Rank>
    public typealias ElementTensor = Rank.ElementTensor

    /// Tensor storage
    internal var base: CoreTensor.Tensor<UnitType>
}

public extension Tensor {
    /// Tensor rank
    var rank: UInt {
        return Rank.rank
    }

    var dynamicShape: TensorShape {
        return base.shape
    }

    /// Tensor shape
    var shape: Shape {
        return Rank.staticShape(from: dynamicShape)
    }

    /// The number of items (atom) in the tensor.
    var unitCount: Int {
        return base.unitCount
    }

    /// The number of items (atoms) per element (subtensor).
    var unitCountPerElement: Int {
        return base.unitCountPerElement
    }

    /// Contiguous storage
    var units: ContiguousArray<UnitType> {
        return base.units
    }

    /// Capacity reserved for element tensors
    var capacity: Int {
        return base.capacity
    }

    /// Initialize a tensor using an existing slice of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: slice of existing elements in row-major order
    internal init(shape: Shape, units: ContiguousArray<UnitType>) {
        self.init(base: CoreTensor.Tensor(shape: Rank.dynamicShape(from: shape), units: units))
    }

    /// Allocate and initialize a tensor to a repeated value
    /// - parameter shape: tensor shape
    /// - parameter repeating: repeated value
    init(shape: Shape, repeating repeatedValue: UnitType) {
        self.init(base: CoreTensor.Tensor(shape: Rank.dynamicShape(from: shape), repeating: repeatedValue))
    }

    /// Allocate and initialize a tensor using the factory function
    /// - parameter shape: tensor shape
    /// - parameter supplier: factory function providing values lazily
    init(shape: Shape, supplier: () -> UnitType) {
        self.init(base: CoreTensor.Tensor(shape: Rank.dynamicShape(from: shape), supplier: supplier))
    }

    /// Initialize a tensor from a sequence of elements in row-major order
    /// - parameter shape: tensor shape
    /// - parameter elements: sequence of elements in row-major order
    /// - parameter vacancySupplier
    init<S: Sequence>(shape: Shape, units: S,
                      vacancySupplier supplier: (() -> UnitType)? = nil)
        where S.Element == UnitType {
            let contiguousSize = Rank.dynamicShape(from: shape).contiguousSize
            var slice = ContiguousArray(units.prefix(contiguousSize))
            /// If elements fewer than required by the shape and supplier is provided
            /// generate new elements using the supplier until vacancy is filled
            if slice.count < contiguousSize, let supplier = supplier {
                slice.reserveCapacity(contiguousSize)
                slice.append(contentsOf: (0..<contiguousSize).map { _ in supplier() })
            }
            self.init(shape: shape, units: slice)
    }

    /// Initialize a tensor from a tensor slice
    /// - parameter slice: tensor slice
    init(_ slice: TensorSlice<Rank>) {
        self.init(shape: slice.shape, units: slice.units)
    }
}

public extension Tensor where Shape == Shape1D {
    /// Initialize a vector from units
    init<C: Collection>(_ units: C) where C.Element == UnitType, C.IndexDistance == Int {
        self.init(shape: (UInt(units.count)), units: units)
    }
}

public extension Tensor {
    func isSimilar<S>(to other: Tensor<S>) -> Bool where S.UnitType == UnitType {
        return base.isSimilar(to: other.base)
    }

    func isIsomorphic<S>(to other: Tensor<S>) -> Bool where S.UnitType == UnitType {
        return base.isIsomorphic(to: other.base)
    }
}

public extension Tensor where Rank.UnitType : Strideable {
    init(shape: Shape, unitsIncreasingFrom lowerBound: UnitType) {
        self.init(base: CoreTensor.Tensor(shape: Rank.dynamicShape(from: shape), unitsIncreasingFrom: lowerBound))
    }
}

public extension Tensor where Rank.UnitType : Strideable, Rank.UnitType.Stride : SignedInteger, Rank.Shape == (UInt) {
    init(scalarElementsIn bounds: CountableRange<UnitType>) {
        self.init(base: CoreTensor.Tensor(scalarElementsIn: bounds))
    }

    init(scalarElementsIn bounds: CountableClosedRange<UnitType>) {
        self.init(base: CoreTensor.Tensor(scalarElementsIn: bounds))
    }
}

public extension Tensor {
    mutating func updateUnit(at index: Int, to newValue: UnitType) {
        base.updateUnit(at: index, to: newValue)
    }
}

public extension Tensor where Rank.UnitType : Numeric {
    mutating func incrementUnit(at index: Int, by newValue: UnitType) {
        base.incrementUnit(at: index, by: newValue)
    }

    mutating func decrementUnit(at index: Int, by newValue: UnitType) {
        base.decrementUnit(at: index, by: newValue)
    }

    mutating func multiplyUnit(at index: Int, by newValue: UnitType) {
        base.multiplyUnit(at: index, by: newValue)
    }
}

public extension Tensor where Rank.UnitType : BinaryInteger {
    mutating func divideUnit(at index: Int, by newValue: UnitType) {
        base.divideUnit(at: index, by: newValue)
    }
}

public extension Tensor where Rank.UnitType : FloatingPoint {
    mutating func divideUnit(at index: Int, by newValue: UnitType) {
        base.divideUnit(at: index, by: newValue)
    }
}

extension Tensor : RandomAccessCollection {
    public typealias Index = Int
    public typealias Element = ElementTensor
    public typealias SubSequence = TensorSlice<Rank>

    /// Access the scalar element or element tensor at an index
    public subscript(index: Int) -> Element {
        get {
            return Rank.element(of: self, at: index)
        }
        set {
            Rank.updateElement(newValue, at: index, in: &self)
        }
    }

    /// Access the subtensor specified by a contiguous range of indices
    public subscript(bounds: Range<Int>) -> SubSequence {
        get {
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            return SubSequence(base: self, bounds: CountableRange(bounds))
        }
        set {
            precondition(indices ~= bounds.lowerBound && indices ~= bounds.upperBound - 1,
                         "Slice indices are out of bounds")
            precondition(newValue.dynamicShape == dynamicShape.dropFirst().prepending(bounds.count),
                         "Shape mismatch")
            base[bounds] = newValue.base
        }
    }

    /// Returns the number of elements
    public var count: Int {
        return base.count
    }

    /// Returns a sequence of indices for elements
    public var indices: CountableRange<Int> {
        return base.indices
    }

    public var startIndex: Int {
        return base.startIndex
    }

    public var endIndex: Int {
        return base.endIndex
    }

    /// Returns the index after the specified one in the current dimension
    public func index(after i: Int) -> Int {
        return base.index(after: i)
    }

    /// Returns the index before the specified one in the current dimension
    public func index(before i: Int) -> Int {
        return base.index(before: i)
    }
}

public extension Tensor {
    func withUnsafeBufferPointer<Result>
        (_ body: (UnsafeBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try base.withUnsafeBufferPointer(body)
    }

    mutating func withUnsafeMutableBufferPointer<Result>
        (_ body: (inout UnsafeMutableBufferPointer<UnitType>) throws -> Result) rethrows -> Result {
        return try base.withUnsafeMutableBufferPointer(body)
    }
}

extension Tensor : TextOutputStreamable {
    public func write<Target>(to target: inout Target) where Target : TextOutputStream {
        base.write(to: &target)
    }
}

extension Tensor : ExpressibleByArrayLiteral {
    public typealias ArrayLiteralElement = Rank.ArrayLiteralElement

    public init(arrayLiteral elements: ArrayLiteralElement...) {
        self = Rank.makeTensor(from: elements)
    }
}

public typealias Tensor1D<T> = Tensor<Rank1<T>>
public typealias Tensor2D<T> = Tensor<Rank2<T>>
public typealias Tensor3D<T> = Tensor<Rank3<T>>
public typealias Tensor4D<T> = Tensor<Rank4<T>>

public typealias Vector<T> = Tensor1D<T>
public typealias Matrix<T> = Tensor2D<T>

public typealias Float1D<T> = Tensor1D<Float>
public typealias Float2D<T> = Tensor2D<Float>
public typealias Float3D<T> = Tensor3D<Float>
public typealias Float4D<T> = Tensor4D<Float>
public typealias Double1D<T> = Tensor1D<Double>
public typealias Double2D<T> = Tensor2D<Double>
public typealias Double3D<T> = Tensor3D<Double>
public typealias Double4D<T> = Tensor4D<Double>
